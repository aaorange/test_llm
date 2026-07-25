"""
适配器合并的步骤
1、通过命令读取三个参数：
    1、基座模型的参数路径
    2、适配器的参数路径
    3、merge之后的保存路径

2、通过PeftModel.from_pretrained  加载基座模型和适配器，得到peft_model
    2.1 先去加载基座模型
    2.2 PeftModel.from_pretrained将基座模型和适配器，加载到同一个peft model

3、peft_model.merge_and_unload() 讲适配器和基座模型参数，进行合并，得到合并后的model

4、model.save_pretrained(), tokenizer.save_pretrained() 将模型和分词器，再保存到同一个目录下

"""
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_model",type=str)
parser.add_argument("--adapter_model",type=str)
parser.add_argument("--merged_path",type=str)

args = parser.parse_args()

base_model = args.base_model
adapter_model = args.adapter_model
merged_path = args.merged_path

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(adapter_model)


from peft import PeftModel
peft_model = PeftModel.from_pretrained(model,adapter_model)


merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(merged_path)

tokenizer.save_pretrained(merged_path)