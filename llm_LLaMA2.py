from transformers import PretrainedConfig
import torch
import torch.nn as nn


class ModelConfig(PretrainedConfig):
    model_type='Tiny-K'
    def __init__(self,
                 dim:int=768, #模型维度
                 n_layers:int =12, #transformer的层数
                 n_heads:int =16, #注意力机制的头数
                 n_kv_heads:int =8, #键值头的数量
                 vocab_size:int =6144, #词汇表大小，通常是训练数据中不同单词的数量加上特殊标记的数量
                 hidden_dim:int=None, #隐藏层维度
                 multiple_of :int=64,
                 norm_eps:float=1e-5, #归一化层的eps
                 max_seq_len:int =512, #最大序列长度
                 dropout:float=0.0, #dropout率，通常在训练过程中使用，以防止过拟合
                 flash_attn:bool=True, #是否使用flash attention，flash attention是一种优化的注意力机制实现，可以提高计算效率和内存使用效率，特别是在处理长序列时
                 **kwargs, #其他参数
                 ):
        self.dim=dim
        self.n_layers=n_layers
        self.n_heads=n_heads
        self.n_kv_heads=n_kv_heads,
        self.vocab_size=vocab_size,
        self.hidden_size=hidden_dim,
        self.multiple_of=multiple_of,
        self.norm_eps=norm_eps,
        self.max_seq_len=max_seq_len,
        self.dropout=dropout,
        self.flash_attn=flash_attn
        super().__init__(**kwargs) #调用父类的构造函数，传递任何额外的参数（如果有的话），以字典的形式传递给父类的构造函数，以确保父类能够正确地处理这些参数

#使用args时，默认使用ModelConfig类的参数

#构建RMSNorm类
#RMSNorm是一种归一化方法，类似于LayerNorm，但它只使用均方根（RMS）来进行归一化，而不使用均值。这种方法在某些情况下可以提高模型的训练稳定性和性能。
#公式为：RMSNorm(x) = x * (weight / sqrt(mean(x^2) + eps))

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        output=self._norm(x.float()).type_as(x)
        return output*self.weight


#构建LLaMA2 attention类
#尝试使用GQA（分组查询注意力）来实现LLaMA2的注意力机制。GQA是一种优化的注意力机制，旨在提高计算效率和内存使用效率，特别是在处理长序列时。

def repeat_kv(x:torch.Tensor,n_rep:int) -> torch.Tensor:
    """重复键值张量以适应分组查询注意力机制。

    Args:
        x (torch.Tensor): 输入的键值张量，形状为 (batch_size, seq_len,n_kv_heads, head_dim)，其中 n_kv_heads 是键值头的数量，head_dim 是每个头的维度。
        n_rep (int): 重复的次数，即每个键值头的数量。因为分组查询注意力机制中，每个查询头对应多个键值头，所以需要将键值张量重复 n_rep 次。
        """

    batch_size,seq_len,n_kv_heads,head_dim=x.size()

    if n_rep==1:
        return x

    #将x的维度调整为 (batch_size, seq_len, n_kv_heads*n_rep, head_dim)，以便进行重复操作
    return (x.unsqueeze(-2).expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim))


#测试
if __name__=='__main__':
    norm=RMSNorm(dim=128,eps=1e-5)
    input_tensor=torch.randn(1,50,128)
    output=norm(input_tensor)

    print(output.shape)

    x=torch.randn(1,50,8,16)
    print(repeat_kv(x,2).shape)
