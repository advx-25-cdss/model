"""
Steps:
1. SFT (Supervised Fine-Tuning)
2. RLFT (Reinforcement Learning Fine-Tuning)
---
*Then the model can be used for inference in production*
3. RLHF (Reinforcement Learning from Human Feedback) -- Data from daily use, pipeline only.
"""
import gc
from typing import Literal

import torch
from datasets import load_dataset
from settings import DATASET_SFT, DATASET_RFT, DATASET_SFT_SPLIT, DATASET_RFT_SPLIT, SAMPLE_NUM, SAMPLE_BATCH, \
    SAMPLE_REWARD, SAMPLE_PENALTY
from train.base import accelerator, tokenizer
from torch.utils.data import Dataset, DataLoader

prompt = {
    'sft': 'You are a medical expert. Now you will be given a medical question, please answer it in detail.',
    'rft': 'You are a medical expert. Now you will be given a medical question, please answer it in detail. You need to wrap your answer concisely in a \\boxed{answer} format.',
}
prompt_suffix = '/think <|im_end|><|im_start|>assistant\n'


sft_dataset = load_dataset(DATASET_SFT, DATASET_SFT_SPLIT)
rft_dataset = load_dataset(DATASET_RFT, DATASET_RFT_SPLIT)

def format_medical_data_sft(sample):
    # This function formats a single example for Qwen3's thinking mode
    # The goal is to make the model learn to produce Complex_CoT internally
    # before generating the Response.
    # Qwen3 often uses <think> and </think> tags for its reasoning steps.
    return {
        "prompt": f"{prompt['sft']}{sample['Question']}",
        "completion": f"<think>{sample['Complex_CoT']}</think>{sample['Response']}"
    }

def format_medical_data_grpo(sample):
    # This function formats a single example for Qwen3's thinking mode
    # The goal is to make the model learn to produce Complex_CoT internally
    # before generating the Response.
    return {
        "prompt": f"<|im_start|>user\n{prompt['rft']} {sample['Open-ended Verifiable Question']}<|im_end|>\n<|im_start|>assistant\n",
        "answer": sample['Ground-True Answer'],
    }

# Inject prompts to dataset
# Apply the formatting function
sft_dataset = sft_dataset.map(format_medical_data_sft)['train'].select(range(128))

# Ensure the 'text' column is the only one used by SFTTrainer

grpo_dataset = rft_dataset.map(format_medical_data_grpo)['train'].select(range(128))

def verifier(model_ans, corr_ans, corr_score=SAMPLE_REWARD, wrong_score=SAMPLE_PENALTY):
    res = []
    for idx, i in enumerate(model_ans):
        try:
            res.append(corr_score if i in corr_ans[idx] else wrong_score)
        except Exception as e:
            print(f"Error in verifier: {e}")
            res.append(wrong_score)
    return res
