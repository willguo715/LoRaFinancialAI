import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, TaskType


model_name = "Qwen/Qwen2.5-7B-Instruct"

# 4bit 量化配置，需安装 bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_base_model():
    """加载分词器和基础模型（4bit 量化）"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    # 有些 Qwen 模型没有 pad_token，统一用 eos_token 作为 pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def add_lora(model: torch.nn.Module):
    """给基座模型挂 LoRA 适配器"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        # Qwen2.5 常见的 LoRA 目标模块
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


if __name__ == "__main__":
    tokenizer, base_model = load_base_model()
    model = add_lora(base_model)