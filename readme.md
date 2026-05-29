# NLP 学习项目

本项目是一个完整的自然语言处理（NLP）学习实践项目，涵盖从基础中文分词、RNN/LSTM、Transformer 到 LLM（类 LLaMA2 架构）从零实现与训练的全流程。主要使用 PyTorch 和 HuggingFace Transformers 生态。

## 项目结构

```
NLP/
├── 分词.ipynb                        # 中文分词与文本预处理
├── rnn.ipynb                         # 循环神经网络（RNN）模型
├── transformer.ipynb                 # Transformer 模型详解
├── 迁移学习.ipynb                    # BERT 迁移学习实践
├── seq2seq与注意力机制案例.ipynb       # Seq2Seq 与注意力机制
├── tranformer_model.py               # 从零实现完整 Transformer（~500行）
├── llm_LLaMA2.py                     # 类 LLaMA2 架构 LLM（RMSNorm、RoPE、GQA、Flash Attention）
├── llm_pretraining_demo.py           # LLM 预训练数据集与流程
├── llm_sft_demo.py                   # LLM 监督微调（SFT）数据集
├── llm_generate_demo.py              # LLM 文本生成（Top-K、Top-P、温度采样）
├── llm训练流程实践.py                # LLM 完整训练流程实践
├── prompt_builder.py                 # Chat/SFT 提示词构建工具
├── BPE_tokenizer.py                  # BPE 分词器训练
├── demo.py                           # HuggingFace 微调示例
├── requirements.txt                  # 项目依赖
├── readme.md                         # 项目说明文档
├── bert-base-chinese/                # BERT 中文预训练模型
├── chinese-sentiment/                # 中文情感分析模型
├── checkpoint/                       # 模型检查点
├── models/                           # 训练好的模型权重
├── tokenizer/                        # 训练好的分词器
└── data/                             # 数据集目录
```

## 文件说明

### 1. 分词.ipynb

**中文分词与文本预处理基础**

主要内容包括：

- **jieba 分词**：精确模式、搜索引擎模式
- **命名实体识别（NER）**：识别文本中的人名、地名、组织机构等实体
- **One-hot 编码**：使用 PyTorch 实现文本的 One-hot 编码
- **Word2Vec**：词向量训练（CBOW 和 Skip-gram 模型原理）
- **Embedding 层**：PyTorch `nn.Embedding` 的使用
- **文本长度规范**：截断（truncation）和填充（padding）操作
- **N-gram 特征**：提取文本的 N-gram 特征
- **TensorBoard 可视化**：词向量可视化

**依赖库**：jieba, torch, fasttext, tensorboard

### 2. rnn.ipynb

**循环神经网络（RNN）模型**

主要内容包括：

- **RNN 基础**：模型结构、计算过程、参数说明
- **RNN API**：PyTorch `nn.RNN` 的使用方法
- **LSTM 模型**：长短期记忆网络原理与实现
- **Bi-LSTM**：双向 LSTM 模型
- **人名分类器案例**：字符级文本处理、序列 padding、模型训练与评估

**依赖库**：torch, pandas, numpy

### 3. transformer.ipynb

**Transformer 模型详解**

涵盖 Transformer 核心组件：

- **Self-Attention 机制**：Scaled Dot-Product Attention 原理
- **Multi-Head Attention**：多头注意力的实现
- **Positional Encoding**：位置编码的原理与多种实现方式
- **Encoder-Decoder 架构**：完整的编码器-解码器结构
- **Layer Normalization**：层归一化
- **残差连接与 FFN**：残差连接和前馈网络

### 4. 迁移学习.ipynb

**BERT 迁移学习实践**

主要内容包括：

- **BERT 模型加载**：使用 HuggingFace 加载预训练 BERT
- **Feature Extraction**：提取 BERT 特征向量
- **Fill-Mask 任务**：完形填空任务实践
- **下游任务微调**：在特定任务上微调 BERT

### 5. seq2seq与注意力机制案例.ipynb

**Seq2Seq 与注意力机制**

主要内容包括：

- **Encoder-Decoder 框架**：序列到序列模型架构
- **注意力机制**：Bahdanau Attention / Luong Attention 实现
- **机器翻译案例**：完整的翻译任务实践
- **Beam Search**：束搜索解码策略

### 6. tranformer_model.py

**从零实现完整 Transformer（约 500 行）**

纯 PyTorch 实现，包含：

- `ModelArgs`：模型参数配置类
- `MultiheadAttention`：多头注意力（支持因果掩码）
- `LayerNorm`：层归一化
- `MLP`：前馈网络（SwiGLU 激活）
- `PositionEncoding`：正弦位置编码
- `EncoderLayer` / `Encoder`：编码器层和编码器
- `DecoderLayer` / `Decoder`：解码器层和解码器
- `Transformer`：完整的 Transformer 模型（支持训练和推理）

### 7. llm_LLaMA2.py

**类 LLaMA2 架构 LLM 从零实现（约 570 行）**

从零实现现代 LLM 的核心组件：

- `ModelConfig`：模型配置（维度、层数、注意力头数、KV 头数、词表大小等）
- `RMSNorm`：RMS 归一化（替代 LayerNorm，更高效）
- `Attention`：分组查询注意力（GQA, Grouped Query Attention） + KV Cache
- `Rotary Position Embedding (RoPE)`：旋转位置编码，支持任意长度扩展
- `MLP`：SwiGLU 激活的前馈网络
- `DecoderLayer`：解码器层（Pre-Norm 架构）
- `Transformer`：完整 LLM 模型，支持自回归生成
- `generate()`：自回归文本生成（支持 Top-K 采样）
- `eval_tokenizer()`：分词器评估工具

### 8. llm_pretraining_demo.py

**LLM 预训练流程**

实现预训练的核心数据流：

- `Pretraindataset`：预训练数据集类，构建输入-标签对（next-token prediction）
- `eval_on_valid_set()`：验证集评估函数
- 数据加载与批处理

### 9. llm_sft_demo.py

**LLM 监督微调（SFT）流程**

实现指令微调的数据处理：

- `SFTDataset`：SFT 数据集类，支持 JSON 列表
- `collate_fn()`：批次整理函数（处理 padding 和 label masking）
- `eval_on_valid_set()`：验证集评估

### 10. llm_generate_demo.py

**LLM 文本生成演示**

实现可控文本生成：

- `generate_eos()`：支持多种解码策略的自回归生成
- **Top-K 采样**：限制候选词数量
- **Top-P（Nucleus）采样**：核采样
- **Temperature**：温度调节
- **Repetition Penalty**：重复惩罚

### 11. llm训练流程实践.py

**LLM 完整训练流程实践**

端到端的 LLM 训练实践：

- `SFTDataset`：SFT 数据加载与处理
- `collate_fn()`：批次整理
- 模型初始化、训练循环、评估的完整流程

### 12. prompt_builder.py

**Chat/SFT 提示词构建工具**

支持多种提示词格式：

- `build_chat_prompt()`：Chat 格式提示词构建（system/user/assistant 角色）
- `build_sft_prompt()`：SFT 指令格式提示词构建（指令/输入/回答）
- `encode_prompt()`：提示词编码与答案提取
- `collate()`：批次整理

### 13. BPE_tokenizer.py

**BPE 分词器训练**

从头训练 Byte-Pair Encoding 分词器：

- `read_texts_from_files()`：读取训练语料
- `create_tokenizer_config()`：创建分词器配置（含特殊 token）
- `train_tokenizer()`：训练 BPE 分词器并保存

### 14. demo.py

**HuggingFace 模型微调示例**

使用 HuggingFace Transformers 进行模型微调：

- `FinetuningModel`：微调模型类（预训练模型 + 分类头）
- `collate_fn()`：数据整理函数

## 环境要求

- **Python**：3.8+
- **深度学习框架**：PyTorch
- **HuggingFace 生态**：transformers, datasets, tokenizers, peft
- **实验追踪**：swanlab
- **其他依赖**：jieba, pandas, numpy, tqdm, python-dotenv

### 一键安装

```bash
pip install -r requirements.txt
```

### 完整依赖

| 库            | 用途                       |
| ------------- | -------------------------- |
| torch         | 深度学习框架               |
| transformers  | HuggingFace 模型加载与使用 |
| datasets      | 数据集加载与处理           |
| tokenizers    | 分词器训练                 |
| peft          | 参数高效微调（LoRA 等）    |
| jieba         | 中文分词                   |
| pandas        | 数据处理                   |
| numpy         | 数值计算                   |
| tqdm          | 进度条                     |
| swanlab       | 实验追踪与可视化           |
| python-dotenv | 环境变量管理               |
| fasttext      | 词向量训练                 |
| tensorboard   | 可视化                     |

## 快速开始

1. 克隆项目到本地
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 打开 Jupyter Notebook 运行交互式代码：
   ```bash
   jupyter notebook
   ```
4. 或直接运行 Python 脚本：
   ```bash
   python transformer_model.py     # 训练从零实现的 Transformer
   python llm_LLaMA2.py           # 训练类 LLaMA2 架构 LLM
   ```

## 学习路径

建议按照以下顺序学习，从基础到进阶：

### 第一阶段：NLP 基础

1. **分词.ipynb** → 掌握中文分词、文本编码、词向量基础
2. **rnn.ipynb** → 学习序列模型，理解 RNN/LSTM 原理和应用

### 第二阶段：现代架构

3. **transformer.ipynb** → 深入理解 Transformer 架构
4. **seq2seq与注意力机制案例.ipynb** → 掌握 Seq2Seq 与注意力机制
5. **tranformer_model.py** → 从零实现完整 Transformer（理论与实践结合）

### 第三阶段：LLM 实战

6. **迁移学习.ipynb** → 掌握 BERT 预训练模型的迁移学习
7. **BPE_tokenizer.py** → 理解并训练 BPE 分词器
8. **llm_LLaMA2.py** → 从零实现类 LLaMA2 架构的现代 LLM
9. **prompt_builder.py** → 理解 Chat/SFT 提示词构建
10. **llm_pretraining_demo.py** → 掌握 LLM 预训练数据流
11. **llm_sft_demo.py** → 掌握指令微调（SFT）流程
12. **llm_generate_demo.py** → 掌握可控文本生成
13. **llm训练流程实践.py** → 端到端 LLM 训练实战

## 模型权重

项目包含以下预训练/微调后的模型权重：

- `bert-base-chinese/`：BERT 中文预训练模型
- `chinese-sentiment/`：中文情感分析模型
- `checkpoint/`：训练过程中的模型检查点
- `models/`：训练完成的模型权重

## 可视化与日志

### SwanLab 实验追踪

项目使用 SwanLab 进行实验追踪与可视化：

### TensorBoard 可视化

启动 TensorBoard 查看词向量可视化：

```bash
tensorboard --logdir=./runs --host 0.0.0.0
```

然后在浏览器中访问 `http://localhost:6006`

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [RMSNorm / Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [jieba 中文分词](https://github.com/fxsjy/jieba)
- [Word2Vec 论文](https://arxiv.org/abs/1301.3781)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 许可证

本项目仅供学习参考使用。
