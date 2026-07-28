from transformers import BitsAndBytesConfig
import torch
quantization_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4', #使用nf4进行4bit量化
    bnb_4bit_use_double_quant=False, #不使用双重量化
    bnb_4bit_compute_dtype=torch.bfloat16  #反量化成torch.bf16,做运算
)



from transformers import AutoModelForCausalLM
model=AutoModelForCausalLM.from_pretrained('model/Qwen3-8B',quantization_config=quantization_config)



from peft import prepare_model_for_kbit_training
#prepare_model_for_kbit_training
#1.将模型内部组件精度升高成fp32
#2.将线性层的梯度关闭，requires_grad设置为False
model=prepare_model_for_kbit_training(model)

from peft import LoraConfig
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset,DatasetDict

#1.1数据加载
data:DatasetDict=load_dataset('json',data_files={'train':'data/psychology_data.jsonl',})
data=data['train'].train_test_split(test_size=0.05)

from typing import Dict,List

def convert_type(examples:Dict[str,List]):
    """
    将数据转换成SFTTrainer所需要的Language Modeling类型，对话格式
    """

    conversation_list:List[List[Dict]]=examples['conversation']

    all_data_messages_list=[]

    for data in conversation_list:
        human_message=data[0]['human']
        assistant_message=data[0]['assistant']

        message_list=[
            {'role':'user','content':human_message},
            {'role':'assistant','content':assistant_message}

        ]
        all_data_messages_list.append(message_list)

    return {'messages':all_data_messages_list}


mapped_data=data.map(convert_type,batched=True,remove_columns=['conversation_id', 'category', 'conversation', 'dataset'])




from peft import LoraConfig,get_peft_model
tokenizer=AutoTokenizer.from_pretrained('model/Qwen3-8B')

#3.构造loraConfig

lora_config=LoraConfig(

    r=16,
    lora_alpha=16,
    target_modules='all_linear',
    lora_dropout=0.05,
    task_type='CAUSAL_LM'
)

#4.获取peft_model
peft_model=get_peft_model(model,lora_config)

from trl.trainer.sft_config import SFTConfig
import os
os.environ['TENSORBOARD_LOGGING_DIR']='./logs/09_QLoRA_demo'

#5.sftconfig实例

config=SFTConfig(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8, #梯度累积步数
    max_steps=300,
    logging_strategy='steps' 
    logging_steps=10,
    report_to='tensorboard',
    learning_rate=3e-4.
    lr_sheduler_type='cosine',
    warmup_steps=0.1,
    eval_strategy='steps',
    eval_steps=50,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    optim='paged_adam_32bit'  #分页优化器
    save_strategy='steps',
    save_steps=50,
    save_total_limit=2,
    output_dir='./finetuned/09_QLoRA_demo',
    bf16=True,
    gradient_checkpointing=False,
    activation_offloading=False,
    max_length=650,
    assistant_only_loss=True,
    chat_template_path='./new_chat_template.jinja'

)

from trl.trainer.sft_trainer import SFTTrainer
from transformers import AutoTokenizer,AutoModelForCausalLM

#6.就构造trainer

trainer=SFTTrainer(
    model=peft_model,
    args=config,
    processing_class=tokenizer,
    train_dataset=mapped_data['train'],
    eval_dataset=mapped_data['test']

)

#7。训练
trainer.train()

#8.保存模型参数，tokenizer相关的参数
trainer.save_model('./finetuned/09_QLoRA_demo')