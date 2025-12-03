import os
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from lora import load_base_model


ADAPTER_PATH = "./lora_finance_qa_adapter"
MAX_NEW_TOKENS = 512


def load_models(adapter_path: str = ADAPTER_PATH) -> Tuple[AutoTokenizer, torch.nn.Module, Optional[torch.nn.Module]]:
    """分别加载纯基座模型和注入 LoRA 的模型"""
    tokenizer, base_model = load_base_model()
    base_model.eval()

    if os.path.isdir(adapter_path):
        _, base_model_for_lora = load_base_model()  # 重新加载一份独立的基座
        lora_model = PeftModel.from_pretrained(base_model_for_lora, adapter_path)
        lora_model.eval()
    else:
        print(f"[警告] 未找到 {adapter_path}，仅使用基座模型。")
        lora_model = None
    return tokenizer, base_model, lora_model


def build_messages(question: str, history: Optional[List[Dict[str, str]]] = None):
    """构造对话上下文，history 需包含 role/content 字段。"""
    history = history or []
    messages = history + [{"role": "user", "content": question}]
    return messages


def generate_answer(
    question: str,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    history: Optional[List[Dict[str, str]]] = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    device = getattr(model, "device", next(model.parameters()).device)

    messages = build_messages(question, history)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return answer


if __name__ == "__main__":
    tokenizer, base_model, lora_model = load_models()
    
    question = "我有三万元闲钱，计划三年后用于子女教育，应该怎样投资？"
    print("用户：", question)
    base_answer = generate_answer(question, base_model, tokenizer)
    print("\n[基座模型回答]")
    print(base_answer)

    if lora_model is not None:
        lora_answer = generate_answer(question, lora_model, tokenizer)
        print("\n[LoRA 微调模型回答]")
        print(lora_answer)
    else:
        print("\n未加载到 LoRA 适配器，仅展示基座模型回答。")

    question = "对于一名程序员来说，拿出五分之一的工资做理财，希望快速看到效果。如何做？"
    print("\n\n用户：", question)
    base_answer = generate_answer(question, base_model, tokenizer)
    print("\n[基座模型回答]")
    print(base_answer)

    if lora_model is not None:
        lora_answer = generate_answer(question, lora_model, tokenizer)
        print("\n[LoRA 微调模型回答]")
        print(lora_answer)
    else:
        print("\n未加载到 LoRA 适配器，仅展示基座模型回答。")

    question = "中国全称叫什么"
    print("\n\n用户：", question)
    base_answer = generate_answer(question, base_model, tokenizer)
    print("\n[基座模型回答]")
    print(base_answer)

    if lora_model is not None:
        lora_answer = generate_answer(question, lora_model, tokenizer)
        print("\n[LoRA 微调模型回答]")
        print(lora_answer)
    else:
        print("\n未加载到 LoRA 适配器，仅展示基座模型回答。")