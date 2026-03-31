#导包
import torch
import torch.nn as nn
import torch.functional as F
import math
from dataclasses import dataclass
from transformers import BertTokenizer

@dataclass #这里是一个数据类，用于存储模型的参数
class ModelArgs:
    n_embed:int #嵌入维度
    n_heads:int #注意力头数
    dim :int #模型维度
    dropout :float #dropout率
    max_len:int #最大序列长度
    vocab_size :int #词表大小
    block_size:int #块的大小
    n_layers:int #层数

class MultiheadAttention(nn.Module):
    def __init__(self,args:ModelArgs,is_casual=False): #args:ModelArgs是一个参数类，is_casual是一个布尔值，表示是否使用因果注意力
        super().__init__()
        assert args.dim // args.n_heads==0 #断言，隐藏维度必须可以被头数整除


        #wq,wk,wv是三个线性层，用于计算查询、键和值


