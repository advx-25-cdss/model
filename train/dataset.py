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

sft_dataset = load_dataset(DATASET_SFT, split=DATASET_SFT_SPLIT)
rft_dataset = load_dataset(DATASET_RFT, split=DATASET_RFT_SPLIT)

def format_medical_data_sft(sample):
    # This function formats a single example for Qwen3's thinking mode
    # The goal is to make the model learn to produce Complex_CoT internally
    # before generating the Response.
    # Qwen3 often uses <think> and </think> tags for its reasoning steps.
    return {
        "prompt": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"{prompt['sft']}{sample['Question']}"},
                {"role": "assistant", "content": f"<think>{sample['Complex_CoT']}</think>{sample['Response']}"}
            ],
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True is the default for Qwen3 and is handled by the format above
        )
    }

def format_medical_data_grpo(sample):
    # This function formats a single example for Qwen3's thinking mode
    # The goal is to make the model learn to produce Complex_CoT internally
    # before generating the Response.
    return {
        "prompt": tokenizer.apply_chat_template(
            f"{prompt['rft']}{sample['Open-ended Verifiable Question']}",
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=True is the default for Qwen3 and is handled by the format above
        ),
        "answer": sample['Ground-True Answer'],
    }

# Inject prompts to dataset
# Apply the formatting function
sft_dataset = sft_dataset.map(format_medical_data_sft)
# Ensure the 'text' column is the only one used by SFTTrainer
sft_dataset = sft_dataset.remove_columns([col for col in sft_dataset.column_names if col != 'text'])

grpo_dataset = rft_dataset.map()

def verifier(model_ans, corr_ans, corr_score=SAMPLE_REWARD, wrong_score=SAMPLE_PENALTY):
    res = []
    for idx, i in enumerate(model_ans):
        try:
            res.append(corr_score if i in corr_ans[idx] else wrong_score)
        except Exception as e:
            print(f"Error in verifier: {e}")
            res.append(wrong_score)
    return res
