import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from settings import MODEL_NAME, INIT_LORA_WEIGHTS, LORA_RANK, LORA_ALPHA, LORA_DROPOUT

accelerator = Accelerator()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

tokenizer.padding_side = 'left'