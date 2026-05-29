import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, BertModel, BertTokenizer
from datasets import load_dataset
import swanlab  # 导入SwanLab
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("SWANLAB_API_KEY")
swanlab.login(api_key)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ... 其他导入 ...
# 加载数据集
train_dataset = load_dataset('csv', data_files='./data/train(1).csv', split='train')
test_dataset = load_dataset('csv', data_files='./data/test(1).csv', split='train')
validation_dataset = load_dataset('csv', data_files='./data/validation.csv', split='train')
print(train_dataset[0:3])

print(len(train_dataset))  # 去获取训练集大小
print(test_dataset)

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
bert_model = BertModel.from_pretrained('./bert-base-chinese')

bert_model = bert_model.to(device)


def collate_fn(data):
    sents = [i['text'] for i in data]  # 从输入数据中提取文本内容，构建一个包含所有文本的列表sents。这个列表将用于后续的分词和编码操作，以便将文本转换为模型可以处理的格式。
    labels = [i['label'] for i in data]  # 从输入数据中提取标签内容，构建一个包含所有标签的列表labels。这个列表将用于后续的训练和评估操作，以便将标签转换为模型可以处理的格式。

    # 使用分词器对文本进行编码，得到输入ID、注意力掩码和token类型ID等信息。batch_encode_plus方法可以同时处理多个文本，并且支持填充和截断操作，以确保输入文本的长度一致。返回的结果是一个包含编码信息的字典，可以直接用于模型的输入。
    data = tokenizer.batch_encode_plus(sents, padding='max_length', truncation=True, return_tensors='pt',
                                       max_length=500, return_length=True)

    input_ids = data['input_ids']  # 从编码结果中提取输入ID，构建一个包含所有输入ID的张量input_ids。这个张量将用于模型的输入，以便进行特征提取和分类等任务。

    attention_mask = data[
        'attention_mask']  # 从编码结果中提取注意力掩码，构建一个包含所有注意力掩码的张量attention_mask。这个张量将用于模型的输入，以便指示模型在处理输入文本时应该关注哪些标记。

    token_type_ids = data[
        'token_type_ids']  # 从编码结果中提取token类型ID，构建一个包含所有token类型ID的张量token_type_ids。这个张量将用于模型的输入，以便区分输入文本中不同句子或段落的标记。

    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_fn)


# 定义微调网络
class FinetuningModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 定义全连接层
        self.fc = nn.Linear(768, 2)  # 使用2分类

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 获取bert模型的输出
        out = bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 经过全连接层
        out = self.fc(
            out.pooler_output)  # 取出pooler_output作为全连接层的输入，pooler_output是BERT模型输出的一个特殊向量，通常用于文本分类等任务。通过将pooler_output传递给全连接层，可以将预训练模型提取的特征映射到具体的分类标签，从而实现文本分类等下游任务。

        return out


# 初始化SwanLab（在训练开始前）
swanlab.init(
    experiment_name="finetune-experiment",  # 实验名称
    description="BERT微调模型训练",  # 实验描述
    project="my-awesome-project",  # 项目名称
    config={  # 记录超参数
        "learning_rate": 2e-5,
        "epochs": 3,
        "batch_size": 8,
        "model": "FinetuningModel",
        "base_model": "BERT"
    }
)

# 模型训练
model = FinetuningModel()
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
criterion = nn.CrossEntropyLoss()

# 冻结预训练模型参数
for param in bert_model.parameters():
    param.requires_grad = False

epochs = 3
model.train()

for epoch in range(epochs):
    start_time = time.time()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                                  drop_last=True, collate_fn=collate_fn)

    total_loss, total_correct, total_samples = 0, 0, 0
    batch_losses = []  # 记录每个batch的损失

    for batch_idx, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        out = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        # 记录batch信息
        total_samples += labels.size(0)
        batch_loss = loss.item() * input_ids.size(0)
        total_loss += batch_loss
        batch_losses.append(loss.item())
        total_correct += (torch.argmax(out, dim=-1) == labels).sum().item()

        # 每N个batch记录一次训练损失
        if batch_idx % 10 == 0:  # 每10个batch记录一次
            swanlab.log({
                "batch_loss": loss.item(),
                "batch_accuracy": (torch.argmax(out, dim=-1) == labels).float().mean().item()
            })

    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    # 记录每个epoch的指标
    swanlab.log({
        "epoch_loss": epoch_loss,
        "epoch_accuracy": epoch_accuracy,
        "epoch_time": epoch_time,
        "learning_rate": optimizer.param_groups[0]['lr']  # 记录学习率
    })

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Loss: {epoch_loss:.4f}, "
          f"Accuracy: {epoch_accuracy:.4f}, "
          f"Time: {epoch_time:.2f}s")

    # 可选：添加验证集评估
    # if val_dataset:
    #     val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
    #     swanlab.log({
    #         "val_loss": val_loss,
    #         "val_accuracy": val_accuracy
    #     })

# 训练完成
swanlab.finish()
print("训练完成！")
