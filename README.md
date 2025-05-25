# Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation
![overview]()

Code for the ACL 2025 main conference paper "Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation". This repo is based on [trl](https://github.com/huggingface/trl).

This readme file is for datasets Helpful Assistant (Bai et al., 2022),
and Reddit Summary (Stiennon et al., 2020). As for Long Form QA (Wu
et al., 2024), please refer to https://github.com/allenai/FineGrainedRLHF.

**Note:** We adapt the code and following instructions from https://github.com/YangRui2015/RiC. Great thanks for their opensourcing the code! :) And you could refer to this link to learn further how to conduct the experiments. We here will omit some unnecessary descriptions for clear.

## 1. Installation
Install the requirements.
```
pip install -r requirements.txt
```
Please also refer to https://github.com/YangRui2015/RiC for more discussions about the version of trl.

## 2. Usage

We first conduct SFT then do PPO or MOPPO to obtain the backbone models.

### 2.1 SFT
* Training a SFT model:
```
cd ./sft
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch sft.py --base_model_name 'meta-llama/Llama-2-7b-hf' --exp_type 'summary' --wandb_name {name_of_the_experiment} 
```

* Merging the SFT model if using lora for finetuning:
```
python3 merge_sft_lora.py --base_model_name 'meta-llama/Llama-2-7b-hf' --lora_name {path_to_the_lora_file} --save_directory {path_to_the_saving_directory}
```

### 2.2 MOPPO (Seek the Backbone models) 
MOPPO optimizes preference weighted rewards using PPO, based on the SFT model in 2.1. And we use MOPPO to obatin the backbone models for 2.4 merging.

The 'preference' variable below is the $\beta$ in the paper. Therefore you could set $\beta \in \{0.6,0.7,0.8\}$ to achieve Bone Soups as described in paper.

Train MOPPO:
```
cd ./ppo
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch morlhf.py --preference 0.5 --base_model_name {path_to_the_sft_model} --reward_names 'harmless,helpful' --exp_type 'assistant' --wandb_name 'morlhf_llamma2'
```
Here, the 'preference' is the preference for the first reward.

### 2.3 PPO (Rewarded Soups)
Rewarded Soups train $N$ PPO models for $N$ reward models.  

Train PPO model for each reward model:
```
cd ./ppo
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ppo.py --reward_name 'harmless' --base_model_name {path_to_the_SFT_model} --exp_type 'assistant' --wandb_name {name_for_this_experiment}
```

### 2.4 Model Merging (make the Soup)

Belows are the bash for running Bone Soups method. We here using Accelerate to equip DP. And `base_model_path1` and `base_model_path2` are two backbone models obtained in 2.2. Note that the `weight1` and `weight2` should be matched to `base_model_path1` and `base_model_path2` respectively. Please also notice the order of `reward_names`. For further details please refer to BoneSoups.py.

```
cd ./ppo

accelerate launch \
    --main_process_port 29501 \
    --num_machines 1 \
    --machine_rank 0 \
    --num_processes 8 \
    --multi_gpu \
    --gpu_ids "0,1,2,3,4,5,6,7" \
    BoneSoups.py --reward_names 'faithful,summary' --base_model_path1 'models/summary_faith_pref_p3_merged'  --base_model_path2 'models/summary_faith_pref_p7_merged'  --base_model_path3 'models/summary_sft_merged' --exp_type 'summary' --wandb_name '' --weight1 0.3 --weight2 0.7
```




## 3. Citation
If you find our work useful for your research, please cite:
```
@article{xie2025bone,
  title={Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation},
  author={Xie, Guofu and Zhang, Xiao and Yao, Ting and Shi, Yunsheng},
  journal={arXiv preprint arXiv:2502.10762},
  year={2025}
}
```
