# NLP 学习项目

本项目包含自然语言处理（NLP）的基础概念和实践代码，主要使用 PyTorch 和 jieba 进行中文文本处理。

## 项目结构

```
NLP/
├── 分词.ipynb          # 中文分词与文本预处理
├── rnn.ipynb           # 循环神经网络（RNN）模型
├── readme.md           # 项目说明文档
├── .gitignore          # Git忽略文件配置
├── runs/               # TensorBoard 可视化日志
└── data/               # 数据集目录
```

## 文件说明

### 1. 分词.ipynb

**中文分词与文本预处理基础**

主要内容包括：
- **jieba 分词**：使用 jieba 库进行中文分词（精确模式、搜索引擎模式）
- **命名实体识别（NER）**：识别文本中的人名、地名、组织机构等实体
- **One-hot 编码**：使用 PyTorch 实现文本的 One-hot 编码
- **Word2Vec**：词向量训练（CBOW 和 Skip-gram 模型原理）
- **Embedding 层**：PyTorch nn.Embedding 的使用
- **文本长度规范**：截断（truncation）和填充（padding）操作
- **N-gram 特征**：提取文本的 N-gram 特征
- **TensorBoard 可视化**：词向量可视化

**依赖库**：
- jieba
- torch
- fasttext
- tensorboard

### 2. rnn.ipynb

**循环神经网络（RNN）模型**

主要内容包括：
- **RNN 基础**：RNN 模型结构、计算过程、参数说明
- **RNN API**：PyTorch nn.RNN 的使用方法
- **LSTM 模型**：长短期记忆网络原理与实现
- **BI-LSTM**：双向 LSTM 模型
- **人名分类器案例**：使用 RNN 实现人名国籍分类
  - 字符级文本处理
  - 序列数据的 padding 处理
  - 模型训练与评估

**依赖库**：
- torch
- pandas
- numpy

## 环境要求

- Python 3.8+
- PyTorch
- jieba
- pandas
- numpy
- fasttext
- tensorboard

## 快速开始

1. 克隆项目到本地
2. 安装依赖库：
   ```bash
   pip install torch jieba pandas numpy fasttext tensorboard
   ```
3. 打开 Jupyter Notebook 运行代码：
   ```bash
   jupyter notebook
   ```

## TensorBoard 可视化

启动 TensorBoard 查看词向量可视化：

```bash
tensorboard --logdir=./runs --host 0.0.0.0
```

然后在浏览器中访问 `http://localhost:6006`

## 学习路径

建议按照以下顺序学习：

1. **分词.ipynb**：掌握中文分词、文本编码、词向量基础
2. **rnn.ipynb**：学习序列模型，理解 RNN/LSTM 原理和应用

## 数据集

项目使用以下数据集：
- 中文文本数据（用于分词和词向量训练）
- 人名分类数据集（用于 RNN 分类任务）

## 参考资料

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [jieba 中文分词](https://github.com/fxsjy/jieba)
- [Word2Vec 论文](https://arxiv.org/abs/1301.3781)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 许可证

本项目仅供学习参考使用。
