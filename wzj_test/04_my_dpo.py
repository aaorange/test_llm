from dataclasses import dataclass
import torch
from transformers import AutoTokenizer
import numpy as np

@dataclass
class DPOConfig:
    train_data_size:int=100
    eval_data_size:int=20
    batch_size:int=1
    gradient_accumulation_steps:int=4
    learning_rate:float=0.1
    warmup_ratio:float=0.1
    beta:float=0.1

    tokenizer_path:str='model/Qwen3-0.6B-Base'
    sft_model_path:str='finetuned/02_sft_demo'
    dataset_path:str='data/ultrafeedback_binarized'

config=DPOConfig()
tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_path)
if tokenizer.padd_token_id is None:
    tokenizer.pad_token=tokenizer.eos_token

def get_train_data(dpo_config:DPOConfig):
    """
    获取训练用的chosen_data,rejected_data
    每一项都是一整段聊天记录对应的token_id列表
    """
    from datasets import load_dataset

    train_data=load_dataset(dpo_config.dataset_path)['train_prefs']
    train_data=train_data.shuffle()
    train_data=train_data.select(range(dpo_config.train_data_size))

    chosen_result=[]
    rejected_result=[]
    for i in range(dpo_config.train_data_size):
        #处理chosen
        message_list=train_data[i]['chosen']
        result=tokenizer.apply_chat_template(message_list,tokenize=True)['input_ids']
        chosen_result.append(result)

        #处理rejected
        message_list=train_data[i]['rejected']
        result=tokenizer.apply_chat_template(message_list,tokenize=True)['input_ids']
        rejected_result.append(result)

    return chosen_result,rejected_result

def get_eval_data(dpo_config:DPOConfig):
    from datasets import load_dataset
    eval_data=load_dataset(dpo_config.dataset_path)['test_prefs']
    eval_data=eval_data.shuffle()
    eval_data=eval_data.select(range(dpo_config.eval_data_size))

    chosen_result=[]
    rejected_result=[]
    for i in range(dpo_config.eval_data_size):
        message_list=eval_data[i]['chosen']
        result=tokenizer.apply_chat_template(message_list,tokenize=True)['input_ids']
        chosen_result.append(result)

        message_list=eval_data[i]['rejected']
        result=tokenizer.apply_chat_template(message_list,tokenize=True)['input_ids']
        rejected_result.append(result)

    return chosen_result,rejected_result

from typing import List
from transformers import PreTrainedTokenizerFast

def _parse_conversation_turns(eos_position:List[int]):
    user_ends=[pos for pos in eos_position[::2]]
    assistant_ends=[pos for pos in eos_position[1::2]]
    return user_ends,assistant_ends

def _set_answer_masks(mask,user_ends,assistant_ends):
    num_user_turns=len(user_ends)
    num_assistant_turns=len(assistant_ends)
    if num_user_turns==num_assistant_turns:
        for user_end,assistant_end in zip(user_ends,assistant_ends):
            answer_start=user_end+5
            answer_end=assistant_end+1
            mask[answer_start:answer_end]=1
    elif num_user_turns==num_assistant_turns+1:
        for user_end,assistant_end in zip(user_ends[:-1],assistant_ends):
            answer_start=user_end+5
            answer_end=assistant_end+1
            mask[answer_start:answer_end]=1
        last_user_end=user_ends[-1]
        last_answer_start=last_user_end+5
        mask[last_answer_start:]=1

def create_answer_mask(labels,tokenizer:PreTrainedTokenizerFast):
    answer_mask=torch.zeros_like(labels)

    eos_token_id=tokenizer.encode('<im_end>')[0]

    for idx,ids in enumerate(labels):
        eos_positions=torch.where(ids==eos_token_id)[0].tolist()
        user_ends,assistant_ends=_parse_conversation_turns(eos_positions)
        _set_answer_masks(answer_mask[idx],user_ends,assistant_ends)
    return answer_mask  

def compute_loss(
        chosen_log_probs,
        rejected_log_probs,
        ref_chosen_log_probs,
        ref_rejected_log_probs,
        beta
        ):
    margin=(chosen_log_probs-rejected_log_probs-(ref_chosen_log_probs-ref_rejected_log_probs))
    loss=-torch.nn.functional.logsigmoid(beta*margin)
    return loss.mean()

def compute_log_probs(logits,labels,assistant_mask):
    log_probs=torch.log_softmax(logits,dim=-1)
    label_token_log_prob=torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    #非assistant回答区域设为0
    masked_label_token_log_prob=assistant_mask*label_token_log_prob

    #把一条回答中所有的token的log probability相加
    return masked_label_token_log_prob.sum(dim=-1)

def cosine_decay(current_batch,total_batch,warmup_ratio,learning_rate):
    warmup_batch=total_batch*warmup_ratio
    if current_batch<warmup_batch:
        #warmup:学习率从0线性升到设定值
        k=learning_rate/warmup_batch
        return k*current_batch

    progress=(current_batch-warmup_batch)/(total_batch-warmup_batch)

    #progress从0到1时，cos(pi*progress)从1到-1
    decay_level=(np.cos(np.pi*progress)+1)*0.5
    return learning_rate*decay_level

def eval_model(model,ref_model,dpo_config:DPOConfig):
    model.eval()
    eval_chosen_data,eval_rejected_data=get_eval_data(dpo_config)
    total_batch=(len(eval_chosen_data)+dpo_config.batch_size-1)//dpo_config.batch_size
    all_batch_loss=[]
    for current_batch in range(total_batch):
        current_chosen_batch_data=eval_chosen_data[
            current_batch*dpo_config.batch_size:
            (current_batch+1)*dpo_config.batch_size
        ]
        max_chosen_length=max(len(sample) for sample in current_chosen_batch_data)

        for sample in current_chosen_batch_data:
            padding_length=max_chosen_length-len(sample)
 




















    























