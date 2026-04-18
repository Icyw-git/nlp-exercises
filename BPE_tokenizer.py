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

random.seed(42) #设置随机数种子为42，以确保在每次运行代码时生成的随机数序列相同，这对于调试和结果复现非常有用，可以帮助开发者获得一致的结果，特别是在涉及随机性操作的情况下，例如数据分割、模型初始化等。


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
def create_tokenizer_config(save_dir:str)->None:  #创建一个函数，接受一个字符串参数save_dir，表示保存配置文件的目录路径，函数没有返回值
    config={  #bpe分词器的配置文件
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

    special_tokens_map={  #特殊标记映射
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]

    }

    with open(os.path.join(save_dir,'special_tokens_map.json'),'w',encoding='uft-8') as f:
        json.dump(special_tokens_map,f,ensure_acsii=False,indent=4)


def train_tokenizer(data_path:str,save_dir:str,vocab_size:int=8192) -> None:
    os.makedirs(save_dir,exist_ok=True)

    #初始化分词器
    tokenizer=Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.nomalizer=NFKC() #NFKC（Normalization Form KC）是一种Unicode标准的文本规范化形式，用于将文本转换为一种标准化的形式，以便在处理文本时能够更一致地比较和匹配字符。NFKC会将一些字符转换为它们的兼容形式，例如将全角字符转换为半角字符，将某些组合字符分解为基本字符等，这有助于提高文本处理的准确性和一致性。
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False) #ByteLevel预分词器用于将输入文本分割成字节级别的单元，特别适用于处理BPE（Byte Pair Encoding）分词器。它能够正确处理文本中的特殊字符和空格，并且在分词过程中保持原始文本的结构和格式，add_prefix_space参数控制是否在输入文本前添加一个空格，这对于某些语言和分词策略可能是必要的，以确保正确的分词结果。
    tokenizer.decoder=decoders.ByteLevel() #ByteLevel解码器用于将分词后的ID序列转换回原始文本，特别适用于处理字节级别的分词器，如BPE（Byte Pair Encoding）。它能够正确处理分词过程中引入的特殊标记和字节级别的编码，使得解码后的文本与原始输入保持一致。


    #配置特殊的token
    special_tokens={
        "<unk>"
        "<s>",
        "</s>",
        "<|im_start|>",
        "<|im_end|>"
    }

    #配置训练器
    trainer=trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2, #min_frequency参数指定了一个token在训练数据中至少出现的次数，只有当一个token在训练数据中出现的次数达到或超过这个值时，它才会被包含在最终的词汇表中。这个参数有助于过滤掉那些在训练数据中非常罕见的token，从而减少词汇表的大小，提高模型的效率和性能。
        show_progress=True, #在训练过程中显示进度条，帮助用户了解训练的进展情况，特别是在处理大型数据集时，这个功能可以提供有用的反馈，让用户知道训练还需要多长时间。
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() #ByteLevel预分词器的初始字母表，包含了所有可能出现在输入文本中的字节级别的字符，这些字符将被用作分词器训练的基础，以便在训练过程中能够正确地处理和分词输入文本中的各种字符。
    )

    #训练tokenizer
    print(f'Training tokenizer with data from{data_path}')
    texts=read_texts_from_files(data_path)
    tokenizer.train_from_iterator(texts,trainer=trainer,length=os.path.getsize(data_path)) #length参数指定了训练数据的总大小（以字节为单位），这有助于训练器在处理大型数据集时更有效地管理内存和计算资源，同时也可以提供更准确的进度反馈，特别是在训练过程中显示进度条时，这个参数可以帮助用户了解训练的进展情况。

    #验证特殊token映射
    try:
        assert tokenizer.token_to_id("<unk>")==0
        assert tokenizer.token_to_id("<s>")==1
        assert tokenizer.token_to_id("</s>")==2
        assert tokenizer.token_to_id("<|im_start|>")==3
        assert tokenizer.token_to_id("<|im_end|>")==4
    except AssertionError as e:
        print("Special token mapping error:", e)
        raise

    #保存tokenizer文件
    tokenizer.save(os.path.join(save_dir,"tokenizer.json"))

    #创建配置文件
    create_tokenizer_config(save_dir)
    print(f'Tokenizer saved to {save_dir}')



