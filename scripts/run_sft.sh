ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file /home/azureuser/llm-training/recipes/accelerate_configs/multi_gpu_custom.yaml --num_processes=1 \
scripts/run_sft.py /home/azureuser/llm-training/recipes/zephyr-7b-beta/sft/config_lora_custom_gpt_sw3_full.yaml