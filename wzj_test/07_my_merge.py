# 导入命令行参数解析模块
from argparse import ArgumentParser
# 创建参数解析器对象
parser=ArgumentParser()
# 添加命令行参数 --base_model，用于指定基座模型路径
parser.add_argument('--base_model',type=str)
# 添加命令行参数 --adapter_model，用于指定LoRA适配器模型路径
parser.add_argument('--adapter_model',type=str)
# 添加命令行参数 --merged_path，用于指定合并后模型的保存路径
parser.add_argument('--merged_path',type=str)
# 解析命令行参数，获取用户输入的参数值
args=parser.parse_args()
# 从解析结果中取出基座模型路径
base_model=args.base_model
# 从解析结果中取出适配器模型路径
adapter_model=args.adapter_model
# 从解析结果中取出合并模型的保存路径
merged_path=args.merged_path
# 导入transformers库：自动模型类和自动分词器类
from transformers import AutoTokenizer,AutoModelForCausalLM
# 从本地路径加载预训练的基座模型（因果语言模型）
model=AutoModelForCausalLM.from_pretrained(base_model)
# 从适配器路径加载分词器（适配器中包含了训练时使用的tokenizer）
tokenizer=AutoTokenizer.from_pretrained(adapter_model)
# 导入PEFT库的PeftModel类，用于加载和管理LoRA适配器
from peft import PeftModel
# 将基座模型和LoRA适配器加载到一起，得到带适配器的peft模型
peft_model=PeftModel.from_pretrained(model,adapter_model)
# 将LoRA适配器的权重合并到基座模型中，并卸载适配器，返回纯基座模型（已融合LoRA权重）
merged_model=peft_model.merge_and_unload()
# 将合并后的模型权重和配置文件保存到指定目录
merged_model.sabe_pretrained(merged_path)
# 将分词器文件（vocab等）保存到同一目录，方便后续直接加载使用
tokenizer.save_pretrained(merged_path)