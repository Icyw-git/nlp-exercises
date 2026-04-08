from transformers import PretrainedConfig

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