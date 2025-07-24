import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from settings import MODEL_NAME, INIT_LORA_WEIGHTS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

def tokenize(text, direct=False, max_length=1024, pad=False, trn=True, device=accelerator.device):
    if direct:
        res = tokenizer(text, return_tensors='pt', padding=True)
    else:
        res = tokenizer(
            text,
            return_tensors='pt',
            truncation=trn,
            max_length=max_length,
            padding='max_length',
        )
    input_ids = res.input_ids
    attn_mask = res.attention_mask
    return input_ids, attn_mask