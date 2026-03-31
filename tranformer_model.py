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

        #计算q,k,v，通过线性层
        xq,xk,xv=self.wq(q),self.wk(k),self.wv(v)

        #拆分为多头注意力
        xq=xq.view(batch_size,seq_len,self.n_heads,self.head_dim)
        xk=xk.view(batch_size,seq_len,self.n_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_heads,self.head_dim)

        #转换维度
        xq=xq.transpose(1,2)
        xk=xk.transpose(1,2)
        xv=xv.transpose(1,2)

        #计算注意力分数
        scores=torch.matmul(xq,xk.transpose(-2,-1))/math.sqrt(self.head_dim)  #这里要除以根号下头的维度，进行缩放，防止数据过大
        if self.is_casual:
            assert hasattr(self,'mask') #断言，确保模型有掩码属性,hasattr()函数用于检查对象是否具有指定的属性

            #添加掩码
            scores=scores+self.mask[:,:,:seq_len,:seq_len]

            #计算softmax
            scores=F.softmax(scores,dim=-1).type_as(xq) #type_as函数用法是保证softmax的输出类型与xq相同

            #注意力的dropout
            scores=self.attn_drpoout(scores)

            #计算注意力输出
            output=torch.matmul(scores,xv)

            #合并多头
            output=output.transpose(1,2).contiguous().view(batch_size,seq_len,-1)

            #最后投影回残差流 这里的残差流是指输入和输出的维度相同，可以直接相加
            output=self.wo(output)
            output=self.resid_dropout(output)
            return output


#layernorm层

class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super().__init__()
        #利用线性矩阵做映射
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
                #归一化
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdin=True)

        #在最后一个维度发生了广播
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

class MLP(nn.Module):
    def __init__(self,dim,hidden_dim,dropout):
        super().__init__()

        #第一层：从输入维度变换至隐藏维度
        self.w1=nn.Linear(dim,hidden_dim,bias=False)
        self.w2=nn.Linear(hidden_dim,dim,bias=False)
        #定义dropout
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.w1(x)
        x=F.relu(x)
        x=self.w2(x)
        x=self.dropout(x)
        return x


#定义编码层
class EncoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()

        #需要两个子层结构
        self.attention_norm=LayerNorm(args.n_embed)

        #Encoder不需要掩码，is_casual=False
        self.attention=MultiheadAttention(args,is_casual=False)
        self.ffn_norm=LayerNorm(args.n_embed)
        self.feed_forward=MLP(args.dim,args.dim,args.dropout)

    def forward(self,x):
        #注意力子层
        x=self.attention_norm(x)
        h=x+self.attention(x,x,x) #残差链接

        #前馈神经网络
        out=h+self.feed_forward(self.fnn_norm(h))


        return out

''' Encoder块'''

class Encoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layers=nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])

        #通过一个层归一化
        self.norm=LayerNorm(args.n_embed)
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,args):
        super().__init__()

        #需要三个子层结构
        self.attention_norm1=LayerNorm(args.n_embed)
        self.mask_attention=MultiheadAttention(args,is_casual=True)

        self.attention_norm2=LayerNorm(args.n_embed)
        self.attention=MultiheadAttention(args,is_casual=False)

        self.ffn_norm=LayerNorm(args.n_embed)
        self.feed_forward=MLP(args.dim,args.dim,args.dropout)

    def forward(self,x,enc_out):
        #第一个子层：掩码多头注意力
        x=self.attention_norm1(x)
        x= x+self.mask_attention(x,x,x)

        #第二个子层：多头注意力，查询来自解码器的输入，键和值来自编码器的输出
        x=self.attention_norm2(x)
        h=x+self.attention(x,enc_out,enc_out)

        #第三个子层：前馈神经网络
        out=h+self.feed_forward(self.ffn_norm(h))
        return out

class Decoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layers=nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.norm=LayerNorm(args.n_embed)

    def forward(self,x,enc_out):
        for layer in self.layers:
            x=layer(x,enc_out)
        return self.norm(x)


class PositionEncoding(nn.Module):
    def __init__(self,args):
        super().__init__()


















