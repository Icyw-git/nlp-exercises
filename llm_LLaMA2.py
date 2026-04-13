from transformers import PretrainedConfig
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
from transformers import PreTrainedModel,AutoTokenizer

from dataclasses import dataclass
from typing import Optional,Any,Tuple
import inspect

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
        self.n_kv_heads=n_kv_heads
        self.vocab_size=vocab_size
        self.hidden_dim=hidden_dim
        self.multiple_of=multiple_of
        self.norm_eps=norm_eps
        self.max_seq_len=max_seq_len
        self.dropout=dropout
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
        dim (int): 这里的dim指head_dim，即每个注意力头的维度。旋转嵌入需要为每个注意力头的特征向量计算旋转角度，因此dim表示每个头的维度。
        end (int): 最大序列长度，即需要预计算的旋转嵌入的数量。
        theta (float, optional): 旋转角度的基数，默认为10000.0。这个值用于计算旋转角度，通常选择一个较大的值以确保旋转角度在合理范围内。
    """

    freqs=1.0/(theta **(torch.arange(0,dim,2)[:(dim//2)].float() / dim)) #对应的公式为：freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:, (dim // 2)].float() / dim))，其中torch.arange(0, dim, 2)生成一个从0到dim-1的偶数序列，表示特征向量中实部的索引。[:, (dim // 2)]将这个序列调整为一个列向量，以便后续计算。然后，将这个列向量除以dim，并使用theta作为基数计算频率。
    t=torch.arange(0,end,device=device) #表示位置索引的序列，从0到end-1，表示需要预计算的旋转嵌入的数量。这个序列将用于计算每个位置的旋转角度。
    freqs=torch.outer(t,freqs).float() #相乘得到一个形状为 (end, dim//2) 的张量，其中每个元素表示位置索引与对应频率的乘积，表示每个位置的旋转角度。
    freqs_cos=torch.cos(freqs)
    freqs_sin=torch.sin(freqs)

    return freqs_cos,freqs_sin


def reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor):
    ndim=x.ndim #获取张量的维度，ndim是输入张量x的维度，表示张量的轴数。例如，如果x是一个形状为 (batch_size, seq_len, n_heads, head_dim) 的张量，那么ndim将等于4。

    assert 0<=1<ndim #确保维度1在张量的维度范围内

    assert freqs_cis.shape==(x.shape[1],x.shape[-1]) #确保freqs_cis的形状与x的形状兼容，freqs_cis的形状应该是 (seq_len, dim)，其中seq_len是x的第二维长度，dim是x的最后一维长度

    #将freqs_cis的形状调整为 (1, seq_len, 1, dim)，以便与输入张量x进行广播操作
    return freqs_cis.unsqueeze(0).unsqueeze(-2) #首先使用unsqueeze(0)在第0维添加一个新的维度，使得张量的形状变为 (1, seq_len, dim)。然后，使用unsqueeze(-2)在倒数第二维添加一个新的维度，使得张量的形状变为 (1, seq_len, 1, dim)，以适应输入张量x的形状，并允许在后续计算中进行广播操作。


#实现旋转嵌入
def rotary_emb(xq:torch.Tensor,xk:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor) ->tuple[torch.Tensor,torch.Tensor]:
    #将旋转嵌入应用至q,k矩阵，使得在进行注意力计算的时候，模型能够利用位置信息来更好地理解序列中元素之间的相对位置关系，从而提高模型的性能。
    #重塑q,k维度，分离实部和虚部
    xq_r,xq_i=xq.float().reshape(xq.shape[:-1]+(-1,2)).unbind(-1) #这里reshape将输入张量xq的最后一个维度调整为2，表示实部和虚部。具体来说，xq.shape[:-1]表示输入张量xq的所有维度，除了最后一个维度。通过在最后一个维度上添加(-1, 2)，我们将最后一个维度调整为2，表示实部和虚部。然后，使用unbind(-1)将这个新的维度分离成两个独立的张量xq_r和xq_i，分别表示实部和虚部。
    xk_r,xk_i=xk.float().reshape(xk.shape[:-1]+(-1,2)).unbind(-1)

    #将频率张量重新塑形
    freqs_cos=reshape_for_broadcast(freqs_cos,xq_r)
    freqs_sin=reshape_for_broadcast(freqs_sin,xq_r)

    #应用旋转公式
    xq_out_r=xq_r*freqs_cos-xq_i*freqs_sin #根据旋转公式，实部的输出xq_out_r是输入张量xq的实部xq_r乘以频率张量freqs_cos减去输入张量xq的虚部xq_i乘以频率张量freqs_sin。这个公式表示了实部在旋转过程中的变化。
    xq_out_i=xq_r*freqs_sin+xq_i*freqs_cos
    xk_out_r=xk_r*freqs_cos-xk_i*freqs_sin
    xk_out_i=xk_r*freqs_sin+xk_i*freqs_cos

    #将最后两个维度合并
    xq_out=torch.stack([xq_out_r,xq_out_i],dim=-1).flatten(3)
    xk_out=torch.stack([xk_out_r,xk_out_i],dim=-1).flatten(3)
    #stack函数将实部和虚部的输出张量沿着最后一个维度进行堆叠，形成一个新的张量，其中实部和虚部被组合在一起。然后，使用flatten(3)将最后两个维度合并成一个维度，以适应后续的计算需求,3表示从第3维开始将后续的维度合并成一个维度。

    return xq_out.type_as(xq),xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self,args:ModelConfig):
        super().__init__()
        #根据是否指定了n_kv_heads确定键值头的数量
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        #确保总头数可以被整除
        assert args.n_heads % self.n_kv_heads ==0

        #模型并行处理大小
        model_parallel_size=1

        #本地计算头数，等于总头数除以模型并行处理大小，本地计算头数是每个设备上实际计算的头数，确保在分布式训练中每个设备上的计算负载均衡
        self.n_local_heads=args.n_heads // model_parallel_size

        #本地键值头数
        self.n_local_kv_heads=self.n_kv_heads // model_parallel_size

        #重复次数n_rep，表示每个查询头对应的键值头的数量，计算公式为总头数除以键值头数
        self.n_rep=self.n_local_heads // self.n_local_kv_heads

        #每个头的维数
        self.head_dim=args.dim // args.n_heads


        #定义权重矩阵
        self.wq=nn.Linear(args.dim,args.n_heads * self.head_dim,bias=False)
        self.wk=nn.Linear(args.dim,self.n_kv_heads *self.head_dim,bias=False)  #wk和wv使用分组查询注意力机制，因此它们的输出维度是键值头的数量乘以每个头的维数，而不是总头数乘以每个头的维数。这是因为在分组查询注意力机制中，每个查询头对应多个键值头，所以键值头的数量通常小于总头数。最后通过广播机制共享键值头
        self.wv=nn.Linear(args.dim,self.n_kv_heads *self.head_dim,bias=False)

        #定义输出矩阵
        self.wo=nn.Linear(args.n_heads* self.head_dim,args.dim,bias=False)

        #定义dropout
        self.attn_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)

        #保存dropout概率
        self.dropout=args.dropout

        #检查是否使用flash attention
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention') #检查torch.nn.functional模块是否具有scaled_dot_product_attention函数，以确定是否可以使用flash attention。如果存在这个函数，说明当前的PyTorch版本支持flash attention，可以利用这个优化的注意力机制来提高计算效率和内存使用效率，特别是在处理长序列时。
        if not self.flash:
            print('当前PyTorch版本不支持flash attention，将使用常规的注意力计算方法。请考虑升级到支持flash attention的版本以获得更好的性能。')

            #创建未来掩码
            mask=torch.full((1,1,args.max_seq_len,args.max_seq_len),float('-inf'))
            mask=torch.triu(mask,diagonal=1)

            #注册为缓冲区
            self.register_buffer('mask',mask)

    def forward(self,x:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):

        batch_size,seq_len,_=x.size()

        #计算q,k,v
        xq=self.wq(x)
        xk=self.wk(x)
        xv=self.wv(x)

        #调整形状
        xq=xq.view(batch_size,seq_len,self.n_local_heads,self.head_dim) #将最后一个维度分离为两个维度

        xk=xk.view(batch_size,seq_len,self.n_local_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_local_kv_heads,self.head_dim)

        #应用旋转嵌入
        xq,xk=rotary_emb(xq,xk,freqs_cos,freqs_sin)

        #将k,v进行维度扩展 以适应分组查询注意力机制
        xk=repeat_kv(xk,self.n_rep)
        xv=repeat_kv(xv,self.n_rep)

        #将seq_len和头数置换，以适应注意力计算的标准维度
        xq=xq.transpose(1,2)
        xk=xk.transpose(1,2)
        xv=xv.transpose(1,2)

        if self.flash:
            output=torch.nn.functional.scaled_dot_product_attention(xq,xk,xv,attn_mask=None,dropout_p=self.dropout if self.training else 0)

        else:
            #手动实现注意力计算
            scores=torch.matmul(xq,xk.transpose(-2,-1)) /math.sqrt(self.head_dim) #计算注意力分数

            #断言mask存在
            assert hasattr(self,'mask')

            scores=scores +self.mask[:,:,:seq_len,:seq_len] #与输入长度对齐
            score=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=torch.matmul(scores,xv)

        output=output.transpose(1,2).contiguous().view(batch_size,seq_len,-1) #将头数和每个头的维数合并回原始的维度，以便进行后续的线性变换和输出。首先，使用transpose(1, 2)将头数维度和序列长度维度交换位置，使得输出张量的形状变为 (batch_size, n_heads, seq_len, head_dim)。然后，使用contiguous()确保张量在内存中是连续的，以便进行后续的view操作。最后，使用view(batch_size, seq_len, -1)将头数维度和每个头的维数合并回原始的维度，使得输出张量的形状变为 (batch_size, seq_len, n_heads * head_dim)，以适应后续的线性变换和输出。

        #投影回输入维度
        output=self.wo(output)
        output=self.resid_dropout(output) #在输出上应用残差连接的dropout，以防止过拟合并提高模型的泛化能力。残差连接是一种常见的技术，在深度神经网络中使用，通过将输入直接添加到输出中来帮助缓解梯度消失问题，并促进更深层次的网络训练。通过在残差连接上应用dropout，可以进一步增强模型的鲁棒性和性能。
        return output


class MLP(nn.Module):
    def __init__(self,dim:int,hidden_dim:int,multiple_of:int,dropout:float):
        super().__init__()

        #如果没有指定隐藏层的维度，则默认为输入维度的4倍
        #然后减少至2/3,最后确保它是multiple_of的倍数，multiple_of是一个整数，表示隐藏层维度必须是这个值的倍数。这通常用于确保模型的计算效率和内存使用效率，因为某些硬件架构在处理特定大小的张量时更高效。

        #减少至2/3是为了控制模型的复杂度，防止过拟合，同时确保模型具有足够的表达能力来捕捉输入数据中的复杂模式。通过将隐藏层维度减少到输入维度的2/3，可以在保持模型性能的同时降低计算成本和内存使用。
        if hidden_dim is None:
            hidden_dim=dim*4
            hidden_dim=int(2*hidden_dim/3)
            hidden_dim=multiple_of*((hidden_dim+multiple_of-1)//multiple_of) #为了确保hidden_dim是multiple_of的倍数，使用了这个公式。首先，将hidden_dim加上multiple_of-1，以确保在除以multiple_of时能够正确地向上取整。然后，使用整数除法将这个值除以multiple_of，得到一个整数，表示hidden_dim需要增加多少个multiple_of才能成为一个倍数。最后，将这个整数乘以multiple_of，得到最终的hidden_dim值，这个值是原始hidden_dim的最小倍数。

        self.w1=nn.Linear(dim,hidden_dim,bias=False)
        self.w2=nn.Linear(hidden_dim,dim,bias=False)
        self.w3=nn.Linear(dim,hidden_dim,bias=False)

        self.dropout=nn.Dropout(dropout)

    def forward(self,x):

        #前向传播
        return self.w2(self.dropout(F.silu(self.w1(x))*self.w3(x))) #在前向传播中，首先通过线性变换w1将输入张量x映射到隐藏层维度，然后应用SILU激活函数（也称为Swish激活函数）来引入非线性。接下来，通过线性变换w3将输入张量x映射到隐藏层维度，并与之前的激活结果进行逐元素乘法操作，以实现门控机制。最后，通过线性变换w2将结果映射回输入维度，并应用dropout以防止过拟合并提高模型的泛化能力。



class DecoderLayer(nn.Module):
    def __init__(self,layer_id:int,args:ModelConfig):
        super().__init__()
        self.dim=args.dim
        self.head_dim=args.dim //args.n_heads #每个注意力头的维度
        self.attention=Attention(args)
        self.feed_forward=MLP(
            dim=self.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,


        )
        self.layer_id=layer_id
        self.attention_norm=RMSNorm(dim=args.dim,eps=args.norm_eps)
        self.ffn_norm=RMSNorm(dim=args.dim,eps=args.norm_eps)

    def forward(self,x:torch.Tensor,freqs_cos:torch.Tensor,freqs_sin:torch.Tensor):

        #进行前向传播
        h= x+self.attention(self.attention_norm(x),freqs_cos,freqs_sin) #这里使用了pre-norm结构，即在进行注意力计算之前先对输入张量x进行归一化处理。通过将输入张量x传递给attention_norm层进行归一化，可以帮助模型更好地稳定训练过程，并提高模型的性能。

        out = h+self.feed_forward(self.ffn_norm(h))
        return out


#构建LLaMA2模型类
class Transformer(PreTrainedModel):
    config_class=ModelConfig
    last_loss: Optional[torch.Tensor] #记录最后一次计算的损失

    def __init__(self,args:ModelConfig=None):
        super().__init__(args)

        self.args=args

        self.vocab_size=args.vocab_size

        self.n_layers=args.n_layers

        self.tok_embedding=nn.Embedding(args.vocab_size,args.dim)

        self.dropout=nn.Dropout(args.dropout)

        self.layers=torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id,args))

        self.norm=nn.LayerNorm(args.dim,eps=args.norm_eps)

        self.output=nn.Linear(args.dim,args.vocab_size,bias=False)


        self.tok_embedding.weight=self.output.weight #共享权重，共享权重是一种常见的技术，在语言模型中，输入的词嵌入和输出的词嵌入通常是相同的，因此可以通过将它们的权重矩阵设置为相同来实现权重共享。这不仅可以减少模型的参数数量，还可以提高模型的训练效率和性能，因为输入和输出之间存在强烈的相关性。通过共享权重，模型可以更有效地学习输入和输出之间的关系，从而提高生成文本的质量和准确性。

        freqs_cos,freqs_sin=precompute_freqs_cis(args.dim//args.n_heads,args.max_seq_len)

        self.register_buffer('freqs_cos',freqs_cos,persistent=False) #注册为缓冲区，表示这些张量不需要被优化器更新，但它们是模型的一部分，并且在保存和加载模型时会被包含在内。persistent=False表示这些缓冲区不会被保存到模型的状态字典中
        self.register_buffer('freqs_sin',freqs_sin,persistent=False)

        self.apply(self._init_weights) #参数的初始化，使用apply方法将_init_weights函数应用于模型的所有子模块，以确保所有的权重和偏置都被正确地初始化。这是一个常见的做法，可以帮助模型更快地收敛并提高性能。

        for pn,p in self.named_parameters(): #遍历模型的所有参数，pn是参数的名称，p是参数的张量。通过named_parameters()方法，可以获取模型中每个参数的名称和对应的张量，以便进行特定的初始化操作。
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'): #endwith方法检查参数名称是否以指定的字符串结尾，这里检查是否以'w3.weight'或'wo.weight'结尾，以确定是否需要对这些特定的权重参数进行特殊的初始化处理。这里对这两个投影至输入维度的线性层进行特殊的初始化的原因是，这些层在模型中起着关键的作用，直接影响模型的输出和性能。通过对这些层进行特殊的初始化，可以帮助模型更快地收敛并提高性能，特别是在训练初期阶段。
                torch.nn.init.normal_(p,mean=0,std=0.02/math.sqrt(2*args.n_layers))

        self.last_loss=None #最后一次的损失值，初始为None，表示尚未计算过损失。在模型的前向传播过程中，如果提供了目标标签（targets），模型将计算交叉熵损失并将其存储在last_loss属性中，以便后续使用或分析。如果没有提供目标标签，last_loss将保持为None，表示当前没有可用的损失值。


        self.OUT=CausalLMOutputWithPast() #CausalLMOutputWithPast是一个数据类，这是transformers库中的类，提供标准化的输出格式，包含生成任务所需的所有信息

        self._no_split_modules=[name for name,_ in self.named_modules()] #_no_split_modules是一个列表，包含模型中所有模块的名称。通过named_modules()方法，可以获取模型中每个模块的名称和对应的模块对象。这个列表可以用于指定在模型并行处理或分布式训练过程中不需要进行切分的模块，以确保这些模块在不同设备之间保持完整。


    def _init_weights(self,module):

        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)

    def forward(self,tokens:torch.Tensor,targets :Optional[torch.Tensor]=None,**kwargs) ->torch.Tensor:

        #kwargs是一个可选的字典参数，可以包含输入张量tokens和目标标签targets。通过使用**kwargs，模型的前向传播函数可以接受任意数量的关键字参数，这些参数可以在函数内部进行处理和使用，以实现更灵活的输入和输出处理方式。

        if 'input_ids' in kwargs:
            tokens=kwargs['input_ids']
        if 'labels' in kwargs:
            targets=kwargs['labels']

        bsz,seqlen=tokens.size()

        h=self.tok_embedding(tokens)
        h=self.dropout(h)

        freqs_cos=self.freqs_cos[:seqlen]
        freqs_sin=self.freqs_sin[:seqlen]

        for layer in self.layers:
            h=layer(h,freqs_cos,freqs_sin)

        h=self.norm(h) #在所有的解码器层处理完输入张量h之后，使用LayerNorm进行归一化处理，以帮助模型更好地稳定训练过程，并提高模型的性能。通过对输出张量h进行归一化，可以确保每个位置的特征向量具有相似的尺度，从而有助于模型更有效地学习输入数据中的模式和关系。


        if targets is not None:
            logits=self.output(h)
            self.last_loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=0,reduction='none')

        #这里对logits进行处理，使用view方法将logits张量的形状调整为 (batch_size * seq_len, vocab_size)，以适应交叉熵损失函数的输入要求。targets张量也被调整为 (batch_size * seq_len) 的形状，以便与logits进行逐元素比较。ignore_index=0表示在计算损失时忽略标签值为0的位置，这通常用于处理填充标记（padding token），因为这些位置不应该对模型的训练产生影响。reduction='none'表示返回每个位置的损失值，而不是对所有位置的损失进行平均或求和，这样可以保留每个位置的损失信息，以便后续分析或使用。
        else :
            logits=self.output(h[:,[-1],:]) #推理时只计算最后一个位置的logits，以提高效率，因为在生成任务中，我们通常只关心当前时间步的输出，而不需要计算整个序列的输出。

            self.last_loss=None

        self.OUT.__setitem__('logits',logits)
        self.OUT.__setitem__('last_loss',self.last_loss)
        #设置输出的logits和last_loss属性，以便在模型的前向传播过程中返回这些信息。通过使用__setitem__方法，可以将logits和last_loss存储在OUT对象中，这个对象是CausalLMOutputWithPast类的实例，提供了一个标准化的输出格式，包含生成任务所需的所有信息。
        return self.OUT




@torch.inference_mode()
#这是一个装饰器，表示在这个函数中不需要计算梯度，这对于推理阶段非常有用，可以节省内存和计算资源，提高推理效率。通过使用@torch.inference_mode()装饰器，模型在执行generate函数时将不会计算梯度，从而加快生成过程并减少内存使用。和torch.no_grad()类似，@torch.inference_mode()还会禁用某些特定于训练的功能，如dropout和batch normalization，以确保在推理阶段模型的行为与训练阶段一致，从而提高生成文本的质量和准确性。
def generate(self,idx,stop_id=None,max_new_tokens=256,temperature=1.0,top_k=None):
    """
            给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
            在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
            """
    #stop_id 的作用是指定一个特殊的标记，当生成的 token 等于这个标记时，生成过程将停止。这通常用于控制生成文本的长度或在特定条件下结束生成过程。例如，在语言模型中，可以使用一个特殊的结束标记（如<eos>）作为 stop_id，当模型生成这个标记时，表示生成文本已经完成，可以停止继续生成新的 token。通过使用 stop_id，可以更好地控制生成文本的结构和内容，提高生成结果的质量和相关性。
    #max_new_tokens 参数指定了在生成过程中最多可以生成多少个新的 token。这是一个限制生成文本长度的参数，确保生成过程不会无限进行下去。通过设置 max_new_tokens，可以控制生成文本的长度，避免生成过长的文本，从而提高生成结果的质量和相关性。
    #temperature 参数控制生成文本的多样性和随机性。较高的 temperature 值会增加生成文本的多样性，使得模型更倾向于选择概率较低的 token，从而产生更多样化的输出。较低的 temperature 值会减少生成文本的多样性，使得模型更倾向于选择概率较高的 token，从而产生更保守和确定性的输出。通过调整 temperature 参数，可以根据需要控制生成文本的风格和内容。
    #top_k 参数控制生成文本的多样性和随机性。它指定了在生成过程中，模型只考虑概率最高的 top_k 个 token 进行采样，从而限制了生成文本的选择范围。
    index=idx.shape[1]
    for _ in range(max_new_tokens):
        idx_cond=idx if idx.size(1) <=self.args.max_seq_len else idx[:, -self.args.max_seq_len:] #进行上下文截断，当序列过长的时候，选择保留最近的max_seq_len个token


        logits=self(idx_cond).logits #self(idx_cond)用法是调用模型的前向传播函数，传入当前的输入序列idx_cond，并获取输出对象中的logits属性。这个logits属性包含了模型在当前输入序列下对下一个 token 的预测分数，这些分数可以用于后续的采样过程来生成新的 token。
        logits=logits[:,-1,:] #只取最后一个位置的logits，logits维度变为 (batch_size, vocab_size)，表示模型对下一个 token 的预测分数。通过选择最后一个位置的logits，我们可以专注于当前时间步的输出，从而提高生成效率和准确性，因为在生成任务中，我们通常只关心当前时间步的输出，而不需要计算整个序列的输出。

        if temperature == 0.0:
            _,idx_next=torch.topk(logits,k=1,dim=-1) #top_k函数有两个参数：k和dim。k参数指定了要返回的最大值的数量，这里设置为1，表示只返回最大的一个值。返回值是一个元组，包含了最大值和对应的索引。通过使用_来忽略最大值，只保留索引idx_next，这个索引表示了在logits张量中具有最高分数的token的位置。这个位置对应于模型预测的下一个 token 的索引，可以用于生成新的 token。
        else:
            logits=logits/temperature #对logits进行温度缩放，缩放的目的是为了控制生成文本的多样性和随机性。通过将logits除以temperature，可以调整模型在生成过程中选择下一个 token 的概率分布。当temperature较高时，模型更倾向于选择概率较低的 token，从而产生更多样化的输出；当temperature较低时，模型更倾向于选择概率较高的 token，从而产生更保守和确定性的输出。
            if top_k is not None:
                v,_ =torch.topk(logits,k=min(top_k,logits.size(-1))) #取出logits中概率最高的top_k个值，v是这些值的张量，_是对应的索引。通过使用min(top_k, logits.size(-1))，确保在top_k大于词汇表大小时不会出现错误，因为logits.size(-1)表示词汇表的大小。
                logits[logits<v[:,[-1]]]=-float('inf') #将logits中不在top_k范围内的值设置为负无穷，这样在后续的softmax计算中，这些值的概率将接近于零，从而限制了生成文本的选择范围，提高生成文本的质量和相关性。
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1) #多项式采样，probs是一个形状为 (batch_size, vocab_size) 的张量，表示每个 token 的概率分布。通过使用torch.multinomial函数，可以根据这个概率分布随机采样一个 token 的索引，num_samples=1表示每个样本只采样一个 token。配合温度控制和top_k过滤，这个采样过程可以生成多样化且相关性较高的文本输出。

        if idx_next==stop_id:
            break

        idx=torch.cat((idx,idx_next),dim=1) #将新生成的 token 的索引 idx_next 连接到当前的输入序列 idx 上，形成一个新的输入序列，以便在下一次迭代中继续生成下一个 token。通过使用 torch.cat 函数，可以沿着指定的维度（这里是 dim=1，即序列长度维度）将 idx 和 idx_next 连接起来，从而更新输入序列以包含新生成的 token。
    return idx[:,index:] #输出新生成的token索引


def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) #AutoTokenizer是transformers库中的一个类，用于自动加载预训练的tokenizer。通过调用from_pretrained方法，并传入tokenizer_path参数，可以从指定的路径加载预训练的tokenizer对象。这个对象包含了用于文本处理和编码的各种功能，如分词、编码、解码等，可以用于后续的文本处理和模型输入准备。
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}") #all_special_ids是一个列表，包含了tokenizer中所有特殊token的ID。这些ID对应于特殊token在词汇表中的位置，可以用于在编码和解码过程中正确处理这些特殊token。例如，在语言模型中，特殊token可能包括开始标记（<s>）、结束标记（</s>）、填充标记（<pad>）等，这些标记在生成文本时具有特定的功能和意义。通过查看all_special_ids，可以了解这些特殊token在词汇表中的位置，从而更好地理解和使用tokenizer的功能。

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]

    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        # add_generation_prompt=True
    ) #apply_chat_template是transformers库中的一个方法，用于将一系列消息（messages）应用到聊天模板中，并生成一个格式化的提示文本（prompt）。这个方法可以根据消息的角色（如system、user、assistant）和内容，按照预定义的模板规则将它们组合成一个适合模型输入的文本格式。通过设置tokenize=False，可以选择是否对生成的提示文本进行分词处理，如果设置为True，则会返回一个分词后的张量，而不是原始文本。add_generation_prompt参数可以控制是否在生成的提示文本中添加一个特定的生成提示，以引导模型在生成过程中更好地理解上下文和预期输出。
    print("Generated prompt:\n", prompt, sep="")


    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt) #通过比较解码后的文本（decoded）与原始提示文本（prompt），可以验证编码和解码过程是否正确地保留了文本的内容和结构。如果两者完全匹配，说明tokenizer在编码和解码过程中没有丢失信息或引入错误，从而确保了文本处理的准确性和一致性。


    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded) #在这里，我们将测试文本test_text编码成input_ids，然后再解码回文本。通过比较解码后的文本与原始测试文本，可以验证tokenizer是否正确处理了特殊token（如<|im_start|>和<|im_end|>），并确保这些特殊token在编码和解码过程中得到了正确的保留和处理。这对于模型在处理包含特殊token的输入时能够正确理解和生成文本非常重要。
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)








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
    print(freqs_cos.shape)
    print(freqs_sin)

#测试reshape_for_broadcast函数
    x = torch.randn(1, 50, 8, 16)
    freqs_cis = torch.randn(50, 16)
    reshaped_freqs_cis = reshape_for_broadcast(freqs_cis, x)
    print(reshaped_freqs_cis.shape)


#测试rotary_emb函数
    xq=torch.randn(1,32,8,16)
    xk=torch.randn(1,32,8,16)
    freqs_cos,freqs_sin=precompute_freqs_cis(dim=16,end=32)
    xq_out,xk_out=rotary_emb(xq,xk,freqs_cos,freqs_sin)
    print(xq_out.shape)
    print(xk_out.shape)
    print(xq_out.shape)

#测试Attention类
    args=ModelConfig()
    attention_model=Attention(args)
    batch_size=1
    seq_len=50
    dim=args.dim
    x=torch.randn(batch_size,seq_len,dim)

    freqs_cos,freqs_sin=precompute_freqs_cis(dim//args.n_heads,seq_len)
    output=attention_model(x,freqs_cos,freqs_sin)

    print(output.shape)


#测试MLP类
    dim=128
    mlp=MLP(128,None,args.multiple_of,args.dropout)
    input_tensor=torch.randn(1,50,128)
    output=mlp(input_tensor)
    print(output.shape)

#测试DecoderLayer类
    args=ModelConfig()
    decoder=DecoderLayer(0,args)

    batch_size=1
    seq_len=50
    dim=args.dim
    x=torch.randn(batch_size,seq_len,dim)
    output=decoder.forward(x,freqs_cos,freqs_sin)
    print(output.shape)


#测试Transformer类
    args=ModelConfig()
    x=torch.randint(0,6144,(1,50))
    model=Transformer(args)
    num_params=sum(p.numel() for p in model.parameters())
    print(f'模型参数数量: {num_params}')
    output=model(x)
    print(output.logits.shape)


#测试tokenizer
    tokenizer_path = "D:/NLP/tokenizer"
    eval_tokenizer(tokenizer_path)