export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=./models
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
set -a; source .env; set +a
if [ -z "$MODEL_NAME" ]; then
  echo "MODEL_NAME is not set. Please set it in the .env file."
  exit 1
fi
CUDA_VISIBLE_DEVICES=0,1,2,3,4 trl vllm-serve --model ${MODEL_NAME} --use-peft
