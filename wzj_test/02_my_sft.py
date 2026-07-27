"""从零手写 Qwen SFT：当前学习进度的整洁版本。"""

from dataclasses import dataclass
import math 
import torch
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

@dataclass
class SFTConfig:
    model_path: str = "model/Qwen3-0.6B-Base"
    dataset_path: str = "data/ultrachat_200k"
    train_data_size: int = 200
    eval_data_size: int = 50
    batch_size: int = 1
    learning_rate: float = 3e-5
    max_length: int = 512
    warm_ratio:float=0.1
    warmup_ratio:float=0.1
    save_dir:str='finetuned/my_sft_model'
    eval_every:int=50
    log_dir:str='logs/my_sft_model'


def get_train_data(config, tokenizer):
    """读取训练数据，并将每条聊天记录转换为 token ID 列表。"""
    dataset_dict = load_dataset(config.dataset_path)
    train_dataset = dataset_dict["train_sft"]
    train_dataset = train_dataset.shuffle(seed=42)
    train_dataset = train_dataset.select(range(config.train_data_size))

    all_token_ids = []
    for sample in train_dataset:
        result = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
        )
        token_ids = result["input_ids"][: config.max_length]
        all_token_ids.append(token_ids)

    return all_token_ids

def get_eval_data(config,tokenizer):
    dataset_dict=load_dataset(config.dataset_path)
    eval_dataset=dataset_dict['test_sft']
    eval_dataset=eval_dataset.shuffle(seed=42)
    eval_dataset=eval_dataset.select(range(config.eval_data_size))
    all_token_ids=[]
    for sample in eval_dataset:
        result=tokenizer.apply_chat_template(
            sample['message'],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False
        )
        token_ids=result['input_ids'][:config.max_length]
        all_token_ids.append(token_ids)
    return all_token_ids
                                     

def collate_batch(samples, tokenizer, device):
    """将不同长度的样本右侧 padding，组成一个 batch tensor。"""
    max_length = max(len(sample) for sample in samples)
    padded_samples = []

    for sample in samples:
        padding_length = max_length - len(sample)
        padded_sample = sample + [tokenizer.pad_token_id] * padding_length
        padded_samples.append(padded_sample)

    return torch.tensor(
        padded_samples,
        dtype=torch.long,
        device=device,
    )


def create_answer_mask(labels, tokenizer):
    """返回 1/0 mask：只让 assistant 回复位置参与 loss。"""
    answer_mask = torch.zeros_like(labels)
    eos_token_id = tokenizer.encode(
        "<|im_end|>",
        add_special_tokens=False,
    )[0]

    for batch_index, ids in enumerate(labels):
        eos_positions = torch.where(ids == eos_token_id)[0].tolist()
        user_ends = eos_positions[::2]
        assistant_ends = eos_positions[1::2]

        for user_end, assistant_end in zip(user_ends, assistant_ends):
            answer_start = user_end + 5
            answer_end = assistant_end + 1
            answer_mask[batch_index, answer_start:answer_end] = 1

        # 最后一段 assistant 回复可能因 max_length 截断而没有结束标记。
        if len(user_ends) == len(assistant_ends) + 1:
            last_answer_start = user_ends[-1] + 5
            answer_mask[batch_index, last_answer_start:] = 1

    return answer_mask


def compute_loss(logits, labels, assistant_mask):
    """计算 assistant 回复 token 的平均负对数概率。"""
    log_probs = torch.log_softmax(logits, dim=-1)
    label_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    token_loss = -label_log_probs
    masked_loss = token_loss * assistant_mask
    return masked_loss.sum() / assistant_mask.sum().clamp_min(1)

@torch.no_grad()
def evaluate_model(model,eval_data,tokenizer,config,device):
    model.eval()
    all_losses=[]
    total_batches=(len(eval_data)+config.batch_size-1)//config.batch_size
    for batch_index in range(total_batches):
        start=batch_index*config.batch_size
        end=start+config.batch_size
        current_batch_data=eval_data[start:end]
        batch_tensor=collate_batch(
            current_batch_data,
            tokenizer,
            device
        )
        input_ids=batch_tensor[:,:-1]
        labels=batch_tensor[:,1:]
        assistant_mask=create_answer_mask(
            labels,
            tokenizer
        )
        logits=model(
            input_ids=input_ids
        ).logits
        loss=compute_loss(logits,labels,assistant_mask)
        all_losses.append(loss.item())
    average_loss=sum(all_losses)/len(all_losses)
    return average_loss

    
def cosine_decay(current_step,total_steps,warmup_raio,peak_learning_rate):
    warmup_steps=max(1,int(total_steps*warmup_raio))
    if current_step<warmup_steps:
        return(
            peak_learning_rate*(current_step+1)/warmup_steps
        )
    progess=(
        (current_step-warmup_steps)/max(1,total_steps-warmup_steps)
    )
    return (
        peak_learning_rate*0.5*(1+math.cos(math.pi*progess))
    )

def main():
    config = SFTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    train_data = get_train_data(config, tokenizer)
    eval_data=get_eval_data(config,tokenizer)
    print(f"已加载 {len(train_data)} 条训练数据，设备：{device}")
    writer=SummaryWriter(log_dir=config.log_dir)
    # 下一轮将在这里编写训练循环：
    # batch → input_ids / labels → mask → logits → loss → backward → step
    total_batches=(
        len(train_data)+config.batch_size-1
    )//config.batch_size

    for batch_index in range(total_batches):

        start=batch_index*config.batch_size
        end=start+config.batch_size
        current_batch_data=train_data[start:end]
        batch_tensor=collate_batch(
            current_batch_data,
            tokenizer,
            device
        )
        input_ids=batch_tensor[:,:-1]
        labels=batch_tensor[:,1:]
        assistant_mask=create_answer_mask(labels,tokenizer)
        model.train()
        logits=model(input_ids=input_ids).logits
        loss=compute_loss(logits,labels,assistant_mask)
        optimizer.zero_grad()
        loss.backward()

        current_lr=cosine_decay(
            current_step=batch_index,
            total_steps=total_batches,
            warmup_raio=config.warmup_ratio,
            peak_learning_rate=config.learning_rate
        )

        optimizer.param_groups[0]['lr']=current_lr
        optimizer.step()
        writer.add_scalar(
            'train/loss',
            loss.item(),
            batch_index+1
        )

        writer.add_scalar(
            'train/learning_rate',
            current_lr,
            batch_index+1
        )

        shoule_eval=(
            (batch_index+1)%config.eval_every==0
        )
        if shoule_eval:
            eval_loss=evaluate_model(
                model=model,
                eval_data=eval_data,
                tokenizer=tokenizer,
                config=config,
                device=device,
            )
            print(f'eval_loss={eval_loss:.4f}')
            writer.add_scalar(
                'eval/loss',
                eval_loss,
                batch_index+1
            )



        #验证结束后必须切回训练模式
        model.train()
        print(
            f'batch={batch_index+1}/{total_batches}'
            f'loss={loss.item():.4f}'
            )
    model.save_pretrained(config.save_dir)
    tokenizer.save_pretrained(config.save_dir)
    writer.close()



if __name__ == "__main__":
    main()
