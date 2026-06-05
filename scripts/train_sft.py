import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from llm_LLaMA2 import ModelConfig, Transformer
import swanlab
import time
from dotenv import load_dotenv
import random
import numpy as np

from src.data.sft_dataset import SFTDataset
from src.data.collate import collate_fn

from functools import partial #这里的作用是创建一个新的函数，这个新函数是原函数的一个变体，已经预先填充了一些参数。通过使用partial，我们可以固定一些参数的值，从而简化函数的调用。例如，在这里我们可以使用partial来创建一个新的collate_fn函数，其中pad_id和label_pad_id已经被固定为特定的值，这样在DataLoader中使用这个新的collate_fn时，就不需要每次都传递这些参数了。

from src.training.config import load_config
cfg=load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Configs', 'sft_scratch.yaml'))


# 设置随机种子，确保结果的可复现性，在涉及随机性的操作中，例如数据分割、模型初始化等，使用相同的随机种子可以获得一致的结果，这对于调试和比较不同实验非常有用。
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

load_dotenv()
api_key = os.getenv("SWANLAB_API_KEY")
swanlab.login(api_key)

tokenizer = AutoTokenizer.from_pretrained('tokenizer')

with open(cfg.data.train_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
train_data = data[:int(len(data) * 0.9)]
val_data = data[int(len(data) * 0.9):]

train_dataset=SFTDataset(train_data, tokenizer, cfg.data.max_length, template=cfg.data.template, from_list=True)
val_dataset=SFTDataset(val_data, tokenizer, cfg.data.max_length, template=cfg.data.template, from_list=True)


train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=partial(collate_fn,pad_id=tokenizer.eos_token_id)) #在DataLoader中使用partial函数创建一个新的collate_fn函数，其中pad_id已经被固定为tokenizer.eos_token_id，这样在每次调用collate_fn时，就不需要再传递pad_id参数了，简化了代码的调用，同时确保了在数据加载过程中使用正确的pad_id进行填充。
val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, collate_fn=partial(collate_fn,pad_id=tokenizer.eos_token_id))


def eval_on_valid_set(model, valid_loader):
    model.eval()
    total_loss, total_tokens = 0, 0

    with torch.no_grad():
        for batch in valid_loader:
            inputs = batch['input_ids'][:, :-1].to(device)
            labels = batch['labels'][:, 1:].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.last_loss.mean()
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
        return total_loss / total_tokens


args = ModelConfig()

swanlab.init(
    project='my-awesome-project',
    experiment='llm-sft-demo',
    tags=['sft', 'transformer'],
    config={
        'epochs': cfg.training.epochs,
        'batch_size': cfg.training.batch_size,
        'learning_rate': cfg.training.learning_rate,
        'model_config': args.__dict__
    }
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(args).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, betas=(0.9, 0.95), weight_decay=0.01)

epochs = cfg.training.epochs
best_loss = float('inf')
patience = 10
pat_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss, total_tokens, start, end = 0, 0, 0, 0
    start = time.time()
    for step, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'][:, :-1].to(device)
        labels = batch['labels'][:, 1:].to(device)

        outputs = model(input_ids, labels=labels)

        loss = outputs.last_loss.mean()  # 模型输出的last_loss是一个张量，包含了每个位置的损失值
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens += labels.numel()
        total_loss += loss.item() * labels.numel()

        if step % 100 == 0:
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'rng_state': torch.get_rng_state(),

                }, f'checkpoint_s{step}_e{epoch}.pth'

            )

    end = time.time()
    epoch_loss = total_loss / total_tokens
    epoch_time = end - start
    swanlab.log({
        'epoch_loss': epoch_loss,
        'epoch_time': epoch_time
    })
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')

    val_loss = eval_on_valid_set(model, val_dataloader)
    swanlab.log({
        'epoch': epoch,
        'valid_loss': val_loss
    })

    if val_loss < best_loss:
        best_loss = val_loss
        pat_counter = 0
    else:
        pat_counter += 1
        if pat_counter > patience:
            print('Early stopping triggered!')
            break

swanlab.finish()
print('训练完成！')
