from unsloth import FastLanguageModel
from trl.trainer.sft_trainer import SFTTrainer
from datasets import load_dataset,DatasetDict

#2.加载量化好的模型
model,tokenizer=FastLanguageModel.from_pretrained('./model/Qwen3-8B',load_in_4bit=True,use_exact_model_name=True,local_files_only=True)

#3.获取peft_model

model=FastLanguageModel.from_pretrained(
    model=model,
    r=16,
    lora_alpha=16,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_dropout=0
)
#4.1加载数据
data:DatasetDict=load_dataset('json',data_files={'train':'data/psychology_data.jsonl'})

data=data['train'].train_test_split(test_size=0.2)

#4.2数据处理
from typing import Dict,List

def convert_type(examples:Dict[str,List]):

    """
    将数据转换成SFTTrainer所需要的Langeage MOdeling类型，对话格式

    """
    conversation_list:List[List[Dict]]=examples['conversation']
    all_data_messages_list=[]
    all_data_text_list=[]
    for data in conversation_list:
        human_message=data[0]['human']
        assistant_message=data[0]['assistant']
        message_list=[
            {'role':'user','content':human_message},
            {'role':'assistant','content':assistant_message}
        ]
        result=tokenizer.apply_chat_template(message_list,tokenize=False)
        all_data_text_list.append(result)

    return {'text':all_data_text_list}

mapped_data=data.map(convert_type,batched=True,remove_columns=['conversation_id','category','conversation','dataset'])


from trl.trainer.sft_config import SFTConfig
import os
os.environ['TENSORBOARD_LOGGING_DIR']='./logs/10_unsloth_demo'

#5.sftconfig实例
config=SFTConfig(
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    max_steps=300,
    logging_strategy='steps',
    logging_steps=10,
    report_to='tensorboard',
    learning_rate=3e-4,
    lr_scheduler_type='cosine',
    warmup_steps=0.1,
    eval_strategy='steps',
    eval_steps=50,
        metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    optim = "paged_adamw_32bit", # 可选择性使用 分页优化器
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    output_dir="./finetuned/10_unsloth_demo",
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length= 650,
)

#6.构trainer
trainer=SFTConfig(
    model=model,
    args=config,
    processing_class=tokenizer,
    train_dataset=mapped_data['train'],
    eval_dataset=mapped_data['test']
)

#7.调用unsloth train_on_reponses_only实现assistant_only_loss
from unsloth.chat_templates import train_on_responses_only

trainer=train_on_responses_only(
    trainer=trainer,
    instruction_part='<|im_start|>user\n',
    response_part='<|im_starat|>assistant\n'
)

#训练与保存
trainer.train()
trainer.save_model('./finetuned/10_unsloth_demo')