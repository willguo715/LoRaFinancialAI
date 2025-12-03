import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from lora import load_base_model, add_lora


DATA_PATH = "finance_qa_mini.json"  # 小型中文问答数据集，由脚本生成
MAX_LENGTH = 1024


class QwenChatDataset(Dataset):
    """将简单的中英文 QA 数据转换为 Qwen 对话格式的数据集"""

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = MAX_LENGTH):
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"{data_path} 不存在，请先生成数据集，或自行准备同格式的 JSON 文件。"
            )
        with open(data_path, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, Any]] = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        messages = example["messages"]

        # 使用 Qwen 自带的 chat_template，将多轮对话拼成模型的输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"][0]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    """简单的 Causal LM collator：按最长序列 padding，并复制 labels"""

    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        # 使用 tokenizer 的 pad_token 进行 padding
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            max_length=None,
            return_tensors="pt",
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


def main():
    # 1. 加载 4bit 量化的基座模型与分词器
    tokenizer, base_model = load_base_model()

    # 2. 挂载 LoRA 适配器
    model = add_lora(base_model)

    # 训练时一般需要关闭 use_cache 以避免梯度检查点冲突
    if getattr(model.config, "use_cache", False):
        model.config.use_cache = False

    # 3. 构建数据集（小型中文问答数据）
    train_dataset = QwenChatDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH)

    # 4. 构建 DataLoader 和数据整理器（padding + label 复制）
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator,
    )

    # 5. 配置优化器与训练超参数（适合小数据+单卡的轻量设置）
    learning_rate = 2e-4
    num_epochs = 2
    gradient_accumulation_steps = 4
    logging_steps = 10

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    device = getattr(model, "device", next(model.parameters()).device)
    model.train()

    global_step = 0
    for epoch in range(num_epochs):
        print(f"===== Epoch {epoch + 1}/{num_epochs} =====")
        running_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            running_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % logging_steps == 0:
                    avg_loss = running_loss / logging_steps
                    print(f"Step {global_step} - loss: {avg_loss:.4f}")
                    running_loss = 0.0

    # 6. 保存 LoRA 适配器权重和 tokenizer
    save_dir = "./lora_finance_qa_adapter"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ LoRA 适配器已保存到: {save_dir}")


if __name__ == "__main__":
    main()


