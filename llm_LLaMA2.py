from transformers import PretrainedConfig
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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
        h= x+self.attention(self.attention_norm(x),freqs_cos,freqs_sin)
        out = h+self.feed_forward(self.ffn_norm(h))
        return out




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



