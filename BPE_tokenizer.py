import random
import json # 用于处理JSON数据
import os
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from tokenizers import (
decoders,
models,
pre_tokenizers,
trainers,
Tokenizer
) # 用于构建和训练BPE分词器

from tokenizers.normalizers import NFKC
from typing import Generator # 用于类型注解，表示生成器类型


#加载训练数据
def read_texts_from_files(file_path:str) ->Generator[str,None,None]:
    with open(file_path,'r',encoding='utf-8') as f:
        for line_num,line in enumerate(f,1):
            try:
                data=json.loads(line) #转换为json类型
                if 'text' not in data:
                    raise KeyError(f'Missing "text" field in line {line_num}')
                yield data ['text'] #生成器，逐行返回文本内容，生成器是一种特殊的迭代器，可以在需要时生成数据，而不是一次性加载所有数据到内存中，这对于处理大型文件非常有用
            except json.JSONDecodeError as e:
                print(f'Error decoding JSON in line {line_num}: {e}')
                continue
            except KeyError as e:
                print(e)
                continue

#创建配置文件
def create_tokenizer_config(save_dir:str)->None:
    config={
         "add_bos_token": False,
         "add_eos_token": False,
         "add_prefix_space": False,
         "bos_token": "<|im_start|>",
         "eos_token": "<|im_end|>",
         "pad_token": "<|im_end|>",
         "unk_token": "<unk>",
         "model_max_length": 1000000000000000019884624838656,
         "clean_up_tokenization_spaces": False,
         "tokenizer_class": "PreTrainedTokenizerFast",
         "chat_template": (
             "{% for message in messages %}"
             "{% if message['role'] == 'system' %}"
             "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
             "{% elif message['role'] == 'user' %}"
             "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
             "{% elif message['role'] == 'assistant' %}"
             "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
             "{% endif %}"
             "{% endfor %}"
             "{% if add_generation_prompt %}"
             "{{ '<|im_start|>assistant\n' }}"
             "{% endif %}")

     }

    with open(os.path.join(save_dir,'tokenizer_config.json'),'w',encoding='uft-8') as f:
         json.dump(config,f,ensure_ascii=False,indent=4) #indent是用来设置JSON文件的缩进格式的参数，设置为4表示每个层级的内容都会缩进4个空格，这样可以使生成的JSON文件更易读和美观

    special_tokens_map={
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]

    }

    with open(os.path.join(save_dir,'special_tokens_map.json'),'w',encoding='uft-8') as f:
        json.dump(special_tokens_map,f,ensure_acsii=False,indent=4)

