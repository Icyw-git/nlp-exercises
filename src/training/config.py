"""
配置管理模块：dataclass 定义 + YAML 加载。
─────────────────────────────────────────────
整体数据流：
   YAML 文件 ──yaml.safe_load()──▶ Python dict ──**解包──▶ dataclass 实例 ──▶ cfg.xxx 使用

@dataclass  = 配置的"骨架"（定义有哪些参数、类型、默认值）
YAML 文件   = 配置的"血肉"（给参数赋具体值）
load_config() = 桥（把 YAML 文本转成 Config 对象）
"""

# ============================================================
# 📌 注释块 0：YAML 编写常见错误 & 正确写法对照
# ============================================================
#
# 【错误 1】冒号后面没有空格
#   ❌ dim:512          → YAML 理解为字符串 "dim:512"，而不是 key=dim, value=512
#   ✅ dim: 512         → 冒号后加空格，正确解析为 key-value
#
# 【错误 2】父子键写在同一行（没有换行缩进）
#   ❌ model: dim:512   → YAML 理解为 model = "dim:512"（一个字符串值）
#   ✅ model:           → 冒号后换行，缩进写子键
#         dim: 512
#
# 【错误 3】缩进混用 Tab 和空格
#   ❌ 用 Tab 缩进       → YAML 只认空格，Tab 会导致解析失败
#   ✅ 用 2 或 4 个空格
#
# 【错误 4】列表写法不兼容
#   ❌ target_modules:['q_proj','k_proj']   → YAML 不解析 Python 风格的列表字符串
#   ✅ target_modules: [q_proj, k_proj]     → YAML 行内数组（方括号+逗号分隔，值不用引号）
#   ✅ target_modules:                      → YAML 多行数组（每行一个 -）
#        - q_proj
#        - k_proj
#
# 完整的正确 YAML 示例 (对应 pretrain.yaml)：
#   model:
#     dim: 512
#     n_layers: 8
#   data:
#     train_path: ./data/input.txt
#     max_length: 512
#   training:
#     learning_rate: 3e-4
#     batch_size: 4
#     epochs: 3
# ============================================================

import yaml
from dataclasses import dataclass, field
from typing import Optional, List


# ============================================================
# 📌 注释块 1：@dataclass 装饰器的用法 & 原理
# ============================================================
#
# @dataclass 是 Python 3.7 引入的装饰器，把它加在类定义上一行，
# Python 会自动帮你生成以下方法：
#
#   ① __init__()       —— 构造函数，参数名 = 你定义的字段名
#   ② __repr__()       —— 打印友好的字符串表示
#   ③ __eq__()         —— 按字段值比较两个实例是否相等
#
# 对比手写类 vs dataclass：
#
#   # ❌ 手写（啰嗦）
#   class ModelConfig:
#       def __init__(self, dim=512, n_layers=8):
#           self.dim = dim
#           self.n_layers = n_layers
#
#   # ✅ dataclass（等价，一行搞定）
#   @dataclass
#   class ModelConfig:
#       dim: int = 512
#       n_layers: int = 8
#
# ── field(default_factory=...) 的作用 ──
#
# Python 不允许用"可变对象"（如 list、dict、另一个类的实例）作为默认值，
# 因为默认值只在类定义时计算一次，所有实例会共享同一个对象：
#
#   ❌ model: ModelConfig = ModelConfig()   # 所有 Config 共享同一个 ModelConfig！
#   ✅ model: ModelConfig = field(default_factory=ModelConfig)
#      # default_factory 指定一个"工厂函数"，每次创建实例时调用它生成新对象
#
# ── ** 解包（字典 → 关键字参数）──
#
# ** 把字典的 key=value 自动展开为函数调用的 参数名=参数值：
#
#   raw = {'dim': 512, 'n_layers': 8}
#   ModelConfig(**raw)
#   # 等价于 ↓
#   ModelConfig(dim=512, n_layers=8)
#
# 前提：字典的 key 名必须和 dataclass 的字段名完全一致！
# ============================================================


@dataclass
class ModelConfig:
    """模型结构配置（从零训练 / LoRA 微调共用）"""
    dim: int = 512               # 隐藏层维度
    n_layers: int = 8            # Transformer 层数
    n_heads: int = 8             # 注意力头数
    vocab_size: int = 51200      # 词表大小
    max_seq_len: int = 512       # 最大序列长度
    model_path: Optional[str] = None  # LoRA 场景：HuggingFace 预训练模型路径


@dataclass
class DataConfig:
    """数据配置"""
    train_path: str = './data/alpaca_data_cleaned.json'  # 训练数据路径
    max_length: int = 512         # 序列最大长度（超出截断，不足填充）
    template: str = 'chatglm2'   # Prompt 模板名称（chatglm2 / qwen2）


@dataclass
class TrainingConfig:
    """训练超参数"""
    learning_rate: float = 3e-4                   # 学习率
    batch_size: int = 4                           # 批次大小
    epochs: int = 4                               # 训练轮数
    fp16: bool = True                             # 是否启用混合精度训练
    gradient_accumulation_steps: int = 4          # 梯度累积步数
    save_steps: int = 500                         # 每隔多少步保存 checkpoint
    eval_steps: int = 500                         # 每隔多少步做一次验证
    logging_steps: int = 1000                     # 每隔多少步记录一次日志
    output_dir: str = './checkpoint'              # 模型和 checkpoint 输出目录


@dataclass
class LoRAConfig:
    """
    LoRA 微调配置（仅 sft_lora 场景使用）
    LoRA = Low-Rank Adaptation，在原始权重旁插入低秩矩阵，
    只训练这些小矩阵，大幅减少可训练参数量。
    """
    r: int = 8                                     # 低秩矩阵的秩（rank），越小参数越少
    lora_alpha: int = 16                           # 缩放因子，实际学习率 = lr * (alpha / r)
    lora_dropout: float = 0.05                     # LoRA 层的 dropout 比例
    target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )  # 要插入 LoRA 的模块名列表。用 lambda 工厂函数确保每个实例独立的列表


# ============================================================
# 总配置容器
# ============================================================
# Config 是"总装箱"——把 ModelConfig / DataConfig / TrainingConfig
# 三个小配置盒装进一个箱子。训练脚本只需要传一个 cfg 对象，
# 就能通过 cfg.model / cfg.data / cfg.training 访问所有配置。
# ============================================================

@dataclass
class Config:
    """总配置——一个对象包圆所有子配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: Optional[LoRAConfig] = None   # 仅 sft_lora 场景有值，从零训练时为 None


# ============================================================
# 📌 注释块 2：load_config() —— YAML 加载原理逐步拆解
# ============================================================
#
# 整个加载过程分三步：
#
# 【Step 1】yaml.safe_load() — 把 YAML 文本解析为 Python 字典
#
#   YAML 文件内容：                Python 字典（raw）：
#   model:                         {
#     dim: 512           →           'model': {'dim': 512, 'n_layers': 8},
#     n_layers: 8                    'data':  {'max_length': 256, ...},
#   data:                            'training': {'epochs': 4, ...}
#     max_length: 256              }
#   training:
#     epochs: 4
#
# 【Step 2】** 解包 — 把字典展开为函数调用的关键字参数
#
#   raw['model'] = {'dim': 512, 'n_layers': 8}
#   ModelConfig(**raw['model'])
#   # 等价于 ↓
#   ModelConfig(dim=512, n_layers=8)
#   #           ↑ 字典 key 自动匹配 dataclass 字段名 ↑
#
#   为什么能自动匹配？因为 YAML 的 key 名（如 dim）和
#   dataclass 的字段名（dim: int = 512）完全一致。
#
# 【Step 3】组装 — 把 3 个子对象装进 Config 总装箱
#
#   Config(
#       model=ModelConfig(**raw['model']),      → cfg.model.dim = 512
#       data=DataConfig(**raw['data']),          → cfg.data.max_length = 256
#       training=TrainingConfig(**raw['training']) → cfg.training.epochs = 4
#   )
#
# ── 训练脚本中的使用方式 ──
#
#   from src.training.config import load_config
#
#   cfg = load_config('configs/sft_scratch.yaml')   # 一行搞定
#
#   model = Transformer(cfg.model).to(device)       # cfg.model 是 ModelConfig 实例
#   optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
#   dataset = SFTDataset(..., max_length=cfg.data.max_length)
#
#   所有原来写死的数字（3e-4, 256, 4...）都换成 cfg.xxx 即可。
# ============================================================

def load_config(yaml_path: str) -> Config:
    """
    从 YAML 文件加载配置，返回 Config 对象。

    参数:
        yaml_path: YAML 配置文件的路径（如 'configs/sft_scratch.yaml'）

    返回:
        Config 对象，可通过 cfg.model / cfg.data / cfg.training 访问子配置
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)  # Step 1: YAML 文本 → Python 字典

    # Step 2+3: 字典解包 → dataclass 实例 → 装进总 Config
    return Config(
        model=ModelConfig(**raw.get('model', {})),
        #        ↑ 如果 YAML 中没写 model 段，用空字典兜底，走默认值
        data=DataConfig(**raw.get('data', {})),
        training=TrainingConfig(**raw.get('training', {})),
        lora=LoRAConfig(**raw['lora']) if 'lora' in raw else None
        #    ↑ LoRA 配置是可选的：YAML 中有 lora 段才创建，否则为 None
    )