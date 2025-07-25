from peft import LoraConfig
from transformers import AutoModelForCausalLM
from settings import MODEL_NAME, INIT_LORA_WEIGHTS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT
import torch

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

peft_config = LoraConfig(
    init_lora_weights=INIT_LORA_WEIGHTS,
    task_type='CAUSAL_LM',
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules='all-linear',
    lora_dropout=LORA_DROPOUT
)
