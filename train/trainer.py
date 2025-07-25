from transformers import TrainingArguments
from trl import SFTConfig, SFTTrainer, GRPOTrainer, GRPOConfig
from settings import TRAIN_STEPS, SAMPLE_TEMPERATURE, SAMPLE_TOP_K, SAMPLE_TOP_P
from train.base import tokenizer
from train.dataset import sft_dataset, grpo_dataset
from train.model import peft_config
from train.model import model


def train():
    def extract_kwargs(**kwargs):
        return kwargs

    # --- 4. Set Up Training Arguments ---
    args = extract_kwargs(
        output_dir="./qwen3_medical_reasoning", # Directory to save checkpoints
        per_device_train_batch_size=1, # Adjust based on your GPU memory (higher is better if possible)
        gradient_accumulation_steps=4, # Accumulate gradients over N steps to simulate larger batch size
        learning_rate=3e-5, # Standard learning rate for fine-tuning
        num_train_epochs=1, # Number of full passes over the dataset
        logging_steps=5,   # Log training metrics every 50 steps
        save_steps=100,     # Save model checkpoint every 500 steps
        bf16=True,          # Enable bfloat16 training precision
        optim="adamw_torch", # Optimizer (you can try "paged_adamw_8bit" with quantization, but "adamw_torch" for bf16)
        report_to="none",   # Disable reporting to external services like Weights & Biases for simplicity
        # You might add evaluation_strategy="steps" and eval_steps if you have a validation set
    )

    sft_trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        peft_config=peft_config, # Pass the LoRA configuration
        args=SFTConfig(**args, eos_token="<|im_end|>"),
    )

    sft_trainer.train()

    def reward_funcs(answer: list[str], **kwargs) -> list[float]:
        result = []
        for i in range(len(answer)):
            try:
                decoded = tokenizer.decode(kwargs["completion_ids"][i], skip_special_tokens=True)
                print(decoded)
                answer_part = answer[i].split('</think>')[1].strip()
                result.append(2.0 if kwargs['answer'][i] in answer_part else -1.0)
            except IndexError:
                result.append(-1.0)
        return result
    
    
    grpo_config = GRPOConfig(
        **args,
        temperature=SAMPLE_TEMPERATURE,
        top_k=SAMPLE_TOP_K,
        top_p=SAMPLE_TOP_P,
        min_p=0.0,
        repetition_penalty=0,
    )
    
    grpo_trainer = GRPOTrainer(
        model=sft_trainer.model,
        train_dataset=grpo_dataset,
        peft_config=peft_config,  # Pass the LoRA configuration
        args=grpo_config,
        reward_funcs=reward_funcs,
    )

    grpo_trainer.train()

    grpo_trainer.save_model('./qwen3_medical_reasoning')