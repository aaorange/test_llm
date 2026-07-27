from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


@dataclass
class DPOConfig:
    # 先用小数据量跑通；原始示例的 main 使用 20_000 条。
    train_data_size: int = 100
    eval_data_size: int = 20
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-6
    warmup_ratio: float = 0.1
    beta: float = 0.1

    tokenizer_path: str = 'model/Qwen3-0.6B-Base'
    sft_model_path: str = 'finetuned/02_sft_demo_backup'
    dataset_path: str = 'data/ultrafeedback_binarized'
    save_dir: str = './finetuned/03_dpo_demo'
    log_dir: str = './logs/03_dpo_demo'
    eval_iter: int = 100
    log_iter: int = 100


config = DPOConfig()
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def _tokenize_messages(messages):
    """兼容不同 transformers 版本的 apply_chat_template 返回类型。"""
    result = tokenizer.apply_chat_template(messages, tokenize=True)
    return result['input_ids'] if isinstance(result, dict) else result


def get_train_data(dpo_config: DPOConfig):
    train_data = load_dataset(dpo_config.dataset_path)['train_prefs']
    train_data = train_data.shuffle()
    train_data = train_data.select(range(dpo_config.train_data_size))

    chosen_result = []
    rejected_result = []
    for i in range(dpo_config.train_data_size):
        chosen_result.append(_tokenize_messages(train_data[i]['chosen']))
        rejected_result.append(_tokenize_messages(train_data[i]['rejected']))
    return chosen_result, rejected_result


def get_eval_data(dpo_config: DPOConfig):
    eval_data = load_dataset(dpo_config.dataset_path)['test_prefs']
    eval_data = eval_data.shuffle()
    eval_data = eval_data.select(range(dpo_config.eval_data_size))

    chosen_result = []
    rejected_result = []
    for i in range(dpo_config.eval_data_size):
        chosen_result.append(_tokenize_messages(eval_data[i]['chosen']))
        rejected_result.append(_tokenize_messages(eval_data[i]['rejected']))
    return chosen_result, rejected_result


def _parse_conversation_turns(eos_positions: List[int]):
    user_ends = eos_positions[::2]
    assistant_ends = eos_positions[1::2]
    return user_ends, assistant_ends


def _set_answer_masks(mask, user_ends, assistant_ends):
    if len(user_ends) == len(assistant_ends):
        pairs = zip(user_ends, assistant_ends)
    elif len(user_ends) == len(assistant_ends) + 1:
        pairs = zip(user_ends[:-1], assistant_ends)
    else:
        raise ValueError('聊天记录的 user/assistant 轮次不符合预期')

    for user_end, assistant_end in pairs:
        # 此 +5 保持原代码的 Qwen 聊天模板假设。
        mask[user_end + 5: assistant_end + 1] = 1

    # 最后一轮 assistant 被截断时，标记其剩余回答区域。
    if len(user_ends) == len(assistant_ends) + 1:
        mask[user_ends[-1] + 5:] = 1


def create_answer_mask(labels, tokenizer: PreTrainedTokenizerFast):
    answer_mask = torch.zeros_like(labels)
    eos_token_id = tokenizer.encode('<|im_end|>')[0]

    for idx, ids in enumerate(labels):
        eos_positions = torch.where(ids == eos_token_id)[0].tolist()
        user_ends, assistant_ends = _parse_conversation_turns(eos_positions)
        _set_answer_masks(answer_mask[idx], user_ends, assistant_ends)
    return answer_mask


def compute_loss(
    chosen_log_probs,
    rejected_log_probs,
    ref_chosen_log_probs,
    ref_rejected_log_probs,
    beta,
):
    margin = (
        chosen_log_probs - rejected_log_probs
        - (ref_chosen_log_probs - ref_rejected_log_probs)
    )
    return -torch.nn.functional.logsigmoid(beta * margin).mean()


def compute_log_probs(logits, labels, assistant_mask):
    log_probs = torch.log_softmax(logits, dim=-1)
    label_token_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    return (assistant_mask * label_token_log_probs).sum(dim=-1)


def cosine_decay(current_batch, total_batch, warmup_ratio, learning_rate):
    warmup_batch = total_batch * warmup_ratio
    if current_batch < warmup_batch:
        return learning_rate * current_batch / warmup_batch

    progress = (current_batch - warmup_batch) / (total_batch - warmup_batch)
    return learning_rate * (np.cos(np.pi * progress) + 1) * 0.5


def make_batch(sequences: Sequence[Sequence[int]]):
    """原代码中 chosen/rejected 重复的 padding、右移和 mask 逻辑。"""
    max_length = max(len(sample) for sample in sequences)
    padded = [
        list(sample) + [tokenizer.pad_token_id] * (max_length - len(sample))
        for sample in sequences
    ]
    data = torch.tensor(padded, dtype=torch.long, device='cuda')
    input_ids = data[:, :-1]
    labels = data[:, 1:]
    padding_mask = torch.where(labels == tokenizer.pad_token_id, 0, 1)
    assistant_mask = create_answer_mask(labels, tokenizer)
    return input_ids, labels, assistant_mask & padding_mask


def pair_log_probs(model, chosen_batch, rejected_batch):
    chosen_input_ids, chosen_labels, chosen_mask = make_batch(chosen_batch)
    rejected_input_ids, rejected_labels, rejected_mask = make_batch(rejected_batch)

    chosen_logits = model(chosen_input_ids).logits
    rejected_logits = model(rejected_input_ids).logits

    chosen_log_prob = compute_log_probs(chosen_logits, chosen_labels, chosen_mask)
    rejected_log_prob = compute_log_probs(
        rejected_logits, rejected_labels, rejected_mask
    )
    return chosen_log_prob, rejected_log_prob


def eval_model(model, ref_model, dpo_config: DPOConfig):
    model.eval()
    eval_chosen_data, eval_rejected_data = get_eval_data(dpo_config)
    total_batch = (len(eval_chosen_data) + dpo_config.batch_size - 1) // dpo_config.batch_size
    all_batch_loss = []

    with torch.no_grad():
        for current_batch in range(total_batch):
            start = current_batch * dpo_config.batch_size
            end = (current_batch + 1) * dpo_config.batch_size

            chosen_batch = eval_chosen_data[start:end]
            rejected_batch = eval_rejected_data[start:end]
            chosen_log_prob, rejected_log_prob = pair_log_probs(
                model, chosen_batch, rejected_batch
            )
            ref_chosen_log_prob, ref_rejected_log_prob = pair_log_probs(
                ref_model, chosen_batch, rejected_batch
            )
            loss = compute_loss(
                chosen_log_prob,
                rejected_log_prob,
                ref_chosen_log_prob,
                ref_rejected_log_prob,
                dpo_config.beta,
            )
            all_batch_loss.append(loss.item())

    return sum(all_batch_loss) / len(all_batch_loss)


def train(dpo_config: DPOConfig):
    model = AutoModelForCausalLM.from_pretrained(dpo_config.sft_model_path).to('cuda')
    ref_model = AutoModelForCausalLM.from_pretrained(dpo_config.sft_model_path).to('cuda')
    model.train()
    ref_model.eval()

    optimizer = AdamW(model.parameters(), lr=dpo_config.learning_rate)
    chosen_data, rejected_data = get_train_data(dpo_config)
    total_batch = (len(chosen_data) + dpo_config.batch_size - 1) // dpo_config.batch_size
    writer = SummaryWriter(log_dir=dpo_config.log_dir)
    progress_bar = tqdm.tqdm(total=total_batch)
    loss_list = []

    for current_batch in range(total_batch):
        start = current_batch * dpo_config.batch_size
        end = (current_batch + 1) * dpo_config.batch_size
        chosen_batch = chosen_data[start:end]
        rejected_batch = rejected_data[start:end]

        # 当前模型的前向需要梯度。
        chosen_log_prob, rejected_log_prob = pair_log_probs(
            model, chosen_batch, rejected_batch
        )
        # 参考模型固定，无需构建反向传播图。
        with torch.no_grad():
            ref_chosen_log_prob, ref_rejected_log_prob = pair_log_probs(
                ref_model, chosen_batch, rejected_batch
            )

        loss = compute_loss(
            chosen_log_prob,
            rejected_log_prob,
            ref_chosen_log_prob,
            ref_rejected_log_prob,
            dpo_config.beta,
        )
        loss_list.append(loss.item())
        (loss / dpo_config.gradient_accumulation_steps).backward()

        current_lr = optimizer.param_groups[0]['lr']
        should_step = (
            (current_batch + 1) % dpo_config.gradient_accumulation_steps == 0
            or current_batch == total_batch - 1
        )
        if should_step:
            current_lr = cosine_decay(
                current_batch,
                total_batch,
                dpo_config.warmup_ratio,
                dpo_config.learning_rate,
            )
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.step()
            optimizer.zero_grad()

        if current_batch % dpo_config.eval_iter == 0:
            eval_loss = eval_model(model, ref_model, dpo_config)
            writer.add_scalar('eval/loss', eval_loss, current_batch)
            model.train()

        if current_batch % dpo_config.log_iter == 0:
            mean_train_loss = sum(loss_list[-dpo_config.log_iter:]) / min(
                len(loss_list), dpo_config.log_iter
            )
            writer.add_scalar('train/loss', mean_train_loss, current_batch)
            writer.add_scalar('train/current_lr', current_lr, current_batch)

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f'{loss.item():.5f}', lr=f'{current_lr:.2e}')

    model.save_pretrained(dpo_config.save_dir)
    tokenizer.save_pretrained(dpo_config.save_dir)
    writer.close()
    print('model and tokenizer saved')


if __name__ == '__main__':
    train(DPOConfig())
