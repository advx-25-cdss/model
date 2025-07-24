import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-32B")
INIT_LORA_WEIGHTS = os.getenv("INIT_LORA_WEIGHTS", 'pissa_niter_32')
LORA_RANK = int(os.getenv("LORA_RANK", 8))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", 16))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.02))
DATASET_SFT = os.getenv("DATASET_SFT", "FreedomIntelligence/medical-o1-reasoning-SFT")
DATASET_RFT = os.getenv("DATASET_RFT", "FreedomIntelligence/medical-o1-verifiable-problem")
DATASET_SFT_SPLIT = os.getenv("DATASET_SFT_SPLIT", "en_mix")
DATASET_RFT_SPLIT = os.getenv("DATASET_RFT_SPLIT", "train")
TRAIN_STEPS = int(os.getenv("TRAIN_STEPS", 100))
SAMPLE_BATCH = int(os.getenv("SAMPLE_BATCH", 16))
SAMPLE_SUB_BATCH = int(os.getenv("SAMPLE_SUB_BATCH", 2))
SAMPLE_NUM = int(os.getenv("SAMPLE_NUM", 4))
SAMPLE_MAX_LENGTH = int(os.getenv("SAMPLE_MAX_LENGTH", 4096))
SAMPLE_TEMPERATURE = float(os.getenv("SAMPLE_TEMPERATURE", 0.1))
SAMPLE_TOP_P = float(os.getenv("SAMPLE_TOP_P", 0.75))
SAMPLE_TOP_K = int(os.getenv("SAMPLE_TOP_K", 50))
SAMPLE_REWARD = float(os.getenv("SAMPLE_REWARD", 2))
SAMPLE_PENALTY = float(os.getenv("SAMPLE_PENALTY", -1))