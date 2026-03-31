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
        self.head_dim=args.dim // args.n_heads #每个注意力头的维度
        self.n_heads=args.n_heads


        #wq,wk,wv是三个线性层，用于计算查询、键和值，实际维度没有改变
        self.wq=nn.Linear(args.dim,self.head_dim*self.n_heads,bias=False)
        self.wk=nn.Linear(args.dim,self.head_dim*self.n_heads,bias=False)
        self.wv=nn.Linear(args.dim,self.head_dim*self.n_heads,bias=False)

        self.wo=nn.Linear(self.head_dim,args.dim,bias=False)

        #dropout层
        self.attn_dropout=nn.Dropout(args.dropout)

        #残差链接的dropout,在最后输出之前使用
        self.resid_dropout=nn.Dropout(args.dropout)

        #创建一个上三角矩阵，作为未来掩码
        #因为是多头注意力，所以要添加一个维度
        if is_casual:# 如果是因果注意力
            mask=torch.full((1,1,args.max_len,args.max_len),float('-inf'))
            mask=torch.triu(mask,diagonal=1) #上三角矩阵，主对角线以上的元素为-inf，其他元素为0

            #将掩码注册为模型的缓冲区
            self.register_buffer('mask',mask)

    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor):
        batch_size,seq_len,_=q.size() #获取批次大小，序列长度




