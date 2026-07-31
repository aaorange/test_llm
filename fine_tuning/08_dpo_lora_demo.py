from peft import LoraConfig
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset, Dataset
import os 
## 1.1 数据加载
train_data:Dataset = load_dataset("./data/ultrafeedback_binarized",split="train_prefs")
test_data:Dataset = load_dataset("./data/ultrafeedback_binarized",split="test_prefs")


train_data = train_data.remove_columns(column_names=['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])
test_data = test_data.remove_columns(column_names=['prompt', 'prompt_id', 'messages', 'score_chosen', 'score_rejected'])



from trl.trainer.dpo_config import DPOConfig

os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs/08_dpo_demo"
config = DPOConfig(
    per_device_train_batch_size=4,
    per_device_eval_batch_size= 4,
    gradient_accumulation_steps=8,
    max_steps=300,
    logging_strategy="steps",
    logging_steps=10,
    report_to="tensorboard",
    # 注意：LoRA微调的学习率，一般来说，会比全参微调，高一个数量级
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    eval_strategy="steps",
    eval_steps=50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    output_dir="./finetuned/08_dpo_demo",
    bf16=True,
    gradient_checkpointing=False,
    max_length= 650,
    # assistant_only_loss=True,
    # chat_template_path="./new_chat_template.jinja"

    # 额外的参数
    beta=0.1
)



from peft import LoraConfig,get_peft_model

model = AutoModelForCausalLM.from_pretrained("finetuned/02_sft_demo")
tokenizer = AutoTokenizer.from_pretrained("finetuned/02_sft_demo")
# 3、构造LoraConfig
lora_config = LoraConfig(
    r = 16,
    lora_alpha = 16,
    # target_modules = ["q_proj","v_proj"]
    # target_modules = ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]
    target_modules = "all-linear",
    lora_dropout = 0.05,
    task_type = "CAUSAL_LM"
)

# 4、获取peft_model
peft_model = get_peft_model(model,lora_config)


from peft import peft_model
from trl.trainer.dpo_trainer import DPOTrainer
trainer = DPOTrainer(
    model=peft_model,
    # ref_model=
    args=config,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer

)

trainer.train()
trainer.save_model("./finetuned/08_dpo_demo")
