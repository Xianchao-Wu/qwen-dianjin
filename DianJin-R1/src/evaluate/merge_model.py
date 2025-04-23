import argparse
import torch
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsdp_path', help="GRPO Checkpoint Path", default='checkpoints/R1-GRPO/global_step_60/actor')
    parser.add_argument('--hf_path', help="HuggingFace Model Path", default='/data/Models/Qwen2.5-7B-Instruct')
    parser.add_argument('--out_path', help="Output Model Path", default='checkpoints/Qwen2.5-7B-Instruct-GRPO')
    parser.add_argument('--world_size', type=int, help="World Size", default=8)
    args = parser.parse_args()

    fsdp_checkpoint_path = args.fsdp_path
    huggingface_model_path = args.hf_path
    output_path = args.out_path
    world_size = args.world_size

    state_dict = defaultdict(list)
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    config = AutoConfig.from_pretrained(huggingface_model_path)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    main()