#通过transformers库完成llm训练和开发

import os

from transformers import AutoConfig
model_path='./models/Qwen2.5-1.5B' #模型下载路径
config=AutoConfig.from_pretrained(model_path) #加载模型配置，AutoConfig是transformers库中的一个类，用于加载预训练模型的配置文件，from_pretrained方法接受一个模型路径作为参数，返回一个包含模型配置的对象，这些配置参数包括模型的层数、隐藏维度、注意力头数等，这些参数将用于定义和训练Transformer模型。
print(config)

from transformers import AutoModelForCausalLM

model=AutoModelForCausalLM.from_pretrained(model_path) #加载模型配置和权重


from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(model_path) #加载模型分词器

print(tokenizer.eos_token_id)
