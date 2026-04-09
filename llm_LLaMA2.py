from transformers import PretrainedConfig
import torch
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    if n_rep==1: #不需要重复，和查询头数量相同，直接返回原始张量
        return x

    #将x的维度调整为 (batch_size, seq_len, n_kv_heads*n_rep, head_dim)，以便进行重复操作，将维度调整为注意力计算的标准维度
    return (x.unsqueeze(-2).expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim).reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim))

    #首先使用unsqueeze(-2)在n_kv_heads维度前添加一个新的维度，使得张量的形状变为 (batch_size, seq_len, n_kv_heads, 1, head_dim)。然后，使用expand方法将这个新添加的维度扩展为n_rep，得到形状为 (batch_size, seq_len, n_kv_heads, n_rep, head_dim)。最后，使用reshape方法将张量重新调整为 (batch_size, seq_len, n_kv_heads*n_rep, head_dim)，以适应分组查询注意力机制中每个查询头对应多个键值头的需求。


#旋转嵌入
#旋转嵌入是一种位置编码方法，旨在为模型提供位置信息。它通过将输入张量中的每个位置的特征向量旋转一定的角度来实现位置编码。这种方法可以帮助模型更好地理解序列中元素之间的相对位置关系，从而提高模型的性能。
#原理：对于输入张量中的每个位置，我们将其特征向量分成两部分：实部和虚部。然后，我们根据位置索引计算一个旋转角度，并将实部和虚部分别旋转这个角度。旋转后的实部和虚部被组合在一起，形成新的特征向量，这个特征向量包含了位置信息。

#公式为：RotaryEmbedding(x) = [x_real * cos(theta) - x_imag * sin(theta), x_real * sin(theta) + x_imag * cos(theta)]，其中theta是根据位置索引计算的旋转角度，x_real和x_imag分别是输入张量中实部和虚部的特征向量。
def precompute_freqs_cis(dim:int,end:int,theta:float=10000.0) -> tuple[torch.Tensor,torch.Tensor]:
    """预计算旋转嵌入的频率。

    Args:
        dim (int): 模型维度，即每个位置的特征向量的维度。
        end (int): 最大序列长度，即需要预计算的旋转嵌入的数量。
        theta (float, optional): 旋转角度的基数，默认为10000.0。这个值用于计算旋转角度，通常选择一个较大的值以确保旋转角度在合理范围内。
    """

    freqs=1.0/(theta **(torch.arange(0,dim,2)[:(dim//2)].float() / dim)) #对应的公式为：freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:, (dim // 2)].float() / dim))，其中torch.arange(0, dim, 2)生成一个从0到dim-1的偶数序列，表示特征向量中实部的索引。[:, (dim // 2)]将这个序列调整为一个列向量，以便后续计算。然后，将这个列向量除以dim，并使用theta作为基数计算频率。
    t=torch.arange(0,end,device=device) #表示位置索引的序列，从0到end-1，表示需要预计算的旋转嵌入的数量。这个序列将用于计算每个位置的旋转角度。
    freqs=torch.outer(t,freqs).float() #相乘得到一个形状为 (end, dim//2) 的张量，其中每个元素表示位置索引与对应频率的乘积，表示每个位置的旋转角度。
    freqs_cos=torch.cos(freqs)
    freqs_sin=torch.sin(freqs)

    return freqs_cos,freqs_sin




#测试
if __name__=='__main__':
#测试RMSNorm类
    norm=RMSNorm(dim=128,eps=1e-5)
    input_tensor=torch.randn(1,50,128)
    output=norm(input_tensor)

    print(output.shape)

#测试repeat_kv函数
    x=torch.randn(1,50,8,16)
    print(repeat_kv(x,2).shape)


#测试precompute_freqs_cis函数
    freqs_cos,freqs_sin=precompute_freqs_cis(dim=128,end=32)
    print(freqs_cos)
    print(freqs_sin)