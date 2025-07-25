import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from settings import MODEL_NAME, INIT_LORA_WEIGHTS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
