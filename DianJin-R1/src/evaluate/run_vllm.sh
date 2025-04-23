CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --served-model-name checkpoints/Qwen2.5-7B-Instruct-GRPO \
  --model checkpoints/Qwen2.5-7B-Instruct-GRPO \
  --tensor_parallel_size 4 \
  --port 8901