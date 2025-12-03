import torch
import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, TaskType
from modelscope import snapshot_download

# 使用 ModelScope 下载模型（国内镜像，更稳定）
model_name = "qwen/Qwen2.5-7B-Instruct"
# 使用绝对路径避免路径问题
CACHE_DIR = os.path.abspath("./models")

# 4bit 量化配置，需安装 bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def load_base_model():
    """加载分词器和基础模型（4bit 量化）"""
    # 使用 ModelScope 下载模型到本地
    print("正在从 ModelScope 下载模型...")
    print(f"缓存目录: {CACHE_DIR}")
    
    try:
        # 使用绝对路径，并设置本地文件系统模式
        model_dir = snapshot_download(
            model_name, 
            cache_dir=CACHE_DIR,
            local_files_only=False,
            revision='master'
        )
        print(f"模型已下载到: {model_dir}")
    except Exception as e:
        print(f"ModelScope 下载失败: {e}")
        print("尝试清理缓存后重新下载...")
        # 清理可能损坏的临时文件
        temp_dir = os.path.join(CACHE_DIR, "._____temp")
        lock_dir = os.path.join(CACHE_DIR, ".lock")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(lock_dir):
            shutil.rmtree(lock_dir, ignore_errors=True)
        # 重试
        model_dir = snapshot_download(
            model_name, 
            cache_dir=CACHE_DIR,
            local_files_only=False,
            revision='master'
        )
        print(f"模型已下载到: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    # 有些 Qwen 模型没有 pad_token，统一用 eos_token 作为 pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def add_lora(model: torch.nn.Module):
    """给基座模型挂 LoRA 适配器"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
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