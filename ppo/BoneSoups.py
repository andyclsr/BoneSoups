from datasets import load_dataset
from transformers import AutoTokenizer

import os
import gc
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from trl import set_seed
import numpy as np
import pandas as pd
from utils import get_clean_data, load_main_tokenizer, save_configs, print_trainable_parameters, \
                  Instructions, Instructions_summary, build_dataset_eval, build_dataset_summary_eval, efficient_merge_weights_with_preference
                  

from multi_reward_models import RewardModels
tqdm.pandas()


# define paths for two datasets
# hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
# summary_dataset_path = 'openai/summarize_from_feedback'

hhrlhf_dataset_path = 'dataset/hh-rlhf'
summary_dataset_path = 'dataset/summarize_from_feedback'
tokenizer_path = 'models/llama2-7b-hf'

@dataclass
class ScriptArguments:
    save_directory: Optional[str] = field(default='./logs_rewardedsoups_summary_eval')
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "minibatch size for eval"})
    wandb_name: Optional[str] = field(default='eval_pposoups_klreg0.2_harmless_helpful', metadata={"help": "Name for this experiment"})
    reward_names:Optional[str] = field(default='harmless,helpful')
    base_model_path1: Optional[str]=field(default='./ppo_llamma2_klreg0.2_harmless/batch_200')
    base_model_path2: Optional[str]=field(default='./ppo_llamma2_klreg0.2_helpful/batch_200')
    base_model_path3: Optional[str]=field(default='')
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    weight1: Optional[float] = field(default=0.5, metadata={"help": "weight for the first model"})
    weight2: Optional[float] = field(default=0.5, metadata={"help": "weight for the second model"})
    frac: Optional[float] = field(default=1.0, metadata={"help": "fraction of the dataset to use"})
    skipk: Optional[bool] = field(default=False, metadata={"help": "skip k search"})
    seed: Optional[int] = field(default=8888, metadata={"help": "seed for random number generator"})
    
    begin: Optional[int] = field(default=0, metadata={"help": "begin index"})
    end: Optional[int] = field(default=11, metadata={"help": "end index"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
base_model_name_1 = script_args.base_model_path1
base_model_name_2 = script_args.base_model_path2
base_model_name_3 = script_args.base_model_path3
tokenier_name = tokenizer_path

# TODO: consider the case when you need to add model3
script_args.weight_mask_rates = [script_args.weight_mask_rate for _ in range(len(script_args.reward_names))]


reward_names = [x.strip() for x in script_args.reward_names.split(',')]
reward_path_tokenizer_dict = {
    'harmless': ['Ray2333/gpt2-large-harmless-reward_model'],
    'helpful': ['Ray2333/gpt2-large-helpful-reward_model'],
    'deberta': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'preference2': ['OpenAssistant/reward-model-deberta-v3-large-v2'],
    'summary': ['Tristan/gpt2_reward_summarization'],
    'faithful':['CogComp/bart-faithful-summary-detector'],
    'humor': ['mohameddhiab/humor-no-humor'],
}
# add prefix in reward_path
for key in reward_path_tokenizer_dict.keys():
    reward_path_tokenizer_dict[key][0] = prefix + reward_path_tokenizer_dict[key][0]

reward_model_path_list = []
rm_tokenizer_path_list = []
for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError
    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)
save_info = {
    'base_model_name_1': base_model_name_1,
    'base_model_name_2': base_model_name_2,
    'base_model_name_3': base_model_name_3,
    'reward_peft_path1': reward_model_path_list[0],
    'reward_peft_path2': reward_model_path_list[1],
    'tokenier_name': tokenier_name
}
for i in range(len(reward_model_path_list)):
    save_info['reward_peft_path{}'.format(i+1)] = reward_model_path_list[i]
save_configs(save_info, os.path.join(script_args.save_directory, script_args.wandb_name))


accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))
reward_models = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id) 


set_seed(script_args.seed)
current_device = Accelerator().local_process_index
print(current_device)


if script_args.frac < 1.0:
    Tfrac = script_args.frac
else:
    Tfrac = None

tokenizer = load_main_tokenizer(tokenier_name)
if exp_type == 'assistant':
    valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test', frac=Tfrac) 
    small_valid_dataset = build_dataset_eval(hhrlhf_dataset_path, tokenizer, reward_models.rm_tokenizers, split='train', frac=0.01) 
    instructions = Instructions()
else:
    valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='test', frac=Tfrac)
    small_valid_dataset = build_dataset_summary_eval(summary_dataset_path, tokenizer, reward_models.rm_tokenizers, split='train', frac=0.1) 
    instructions = Instructions_summary()
print(f"Size of the validation set: {len(valid_dataset)}")

# only remove when the key is in the dataset
if 'prompt' in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns('prompt')
    small_valid_dataset = small_valid_dataset.remove_columns('prompt')
if 'query' in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns('query')
    small_valid_dataset = small_valid_dataset.remove_columns('query')

for key in ['key', 'text', 'response']:
    if key in valid_dataset.column_names:
        valid_dataset = valid_dataset.remove_columns(key)
        small_valid_dataset = small_valid_dataset.remove_columns(key)

def evaluate_model(model, tokenizer, valid_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    valid_data_loader = DataLoader(valid_dataset, batch_size=script_args.mini_batch_size, drop_last=True, collate_fn=data_collator)
    model.resize_token_embeddings(len(tokenizer)) #TODO:?????

    accelerator = Accelerator()
    valid_data_loader = accelerator.prepare(valid_data_loader)
    model.to(accelerator.device)
    model.eval()
    # model, valid_data_loader = accelerator.prepare(model, valid_data_loader)

    generation_kwargs = {
        "max_new_tokens": 128 if exp_type == 'assistant' else 48, 
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9, 
        "do_sample": True,
    }

    full_responses = []
    full_prompts = []
    pbar = tqdm(total=len(valid_dataset) // script_args.mini_batch_size // accelerator.num_processes)
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            # mention here
            response_tensors = accelerator.unwrap_model(model).generate(batch["input_ids"], **generation_kwargs) #length_sampler=output_length_sampler, 
            full_responses.extend(response_tensors)
            full_prompts.extend(batch['input_ids'])
            pbar.update(1)
    
    full_responses = tokenizer.batch_decode(full_responses)
    full_prompts = tokenizer.batch_decode(full_prompts)
    # clean data
    full_prompts, full_responses = get_clean_data(full_responses, full_prompts)

    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in full_responses
    ]
    if hasattr(instructions, 'get_post'):
        rewards_list = reward_models.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_models.get_reward_model_scores(queries_responses)
    
    ### error here may because of old version of transformers/accelerate/peft
    all_rewards = []
    for i in range(reward_models.num_rewards):
        all_rewards.append(accelerator.gather_for_metrics(rewards_list[i]))
    all_full_prompts = accelerator.gather_for_metrics(full_prompts)
    all_full_responses = accelerator.gather_for_metrics(full_responses)
    return all_rewards, all_full_prompts, all_full_responses


print("Evaluating........")
tokenizer.padding_side = "left"
## preference list
M = 10
preferences = np.zeros((M+1, 3))
preferences[:, 0] = np.arange(0,1+ 1/M,1 / M)
preferences[:, 1] = 1 - preferences[:, 0]
preferences = np.round(preferences, 1)

preferences_org = np.arange(0,1+ 1/M,1 / M)

w11 = script_args.weight1
w12 = 1 - w11
w21 = script_args.weight2
w22 = 1 - w21

a = [w11, w21]
b = [w12, w22]
solution1 = np.linalg.solve(np.array([a, b]), [1,0])
solution2 = np.linalg.solve(np.array([a, b]), [0,1])


# here get expo 's params k1 and k2
if script_args.skipk == False:
    a_candidates= [0.1,0.2,0.3,0.4,0.5]
    result1 = []
    max_reward = -1000
    k1 = 0.1
    for a in a_candidates:
        k = 1/(1-a)
        x1 = solution1[0]*k
        x2 = solution1[1]*k
        x3 = 1-k     
        temp_save_path = '/home/temp_models/{}'.format(script_args.wandb_name)
        preference = [x1, x2, x3]
        base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
        merged_model = efficient_merge_weights_with_preference(base_model_list, preference, temp_save_path)

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        print(a, k, preference)
        all_rewards, all_full_prompts, all_full_responses = evaluate_model(merged_model, tokenizer, small_valid_dataset)
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_result = {}
        temp_list =[]
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            temp_list.append(np.mean(evaluation_result['obtained_score{}'.format(i+1)]))
        result1.append(temp_list)
        if temp_list[0] > max_reward:
            max_reward = temp_list[0]
            k1 = k
    if process_id == 0:
        # save results to pickle
        import pickle
        with open(os.path.join(script_args.save_directory, script_args.wandb_name, 'result1.pkl'), 'wb') as f:
            pickle.dump(result1, f)

    a_candidates= [0.1,0.2,0.3,0.4,0.5]
    result2 = []
    k2= 0.1
    max_reward = -1000
    for a in a_candidates:
        k = 1/(1-a)
        x1 = solution2[0]*k
        x2 = solution2[1]*k
        x3 = 1-k
        temp_save_path = '/home/temp_models/{}'.format(script_args.wandb_name)
        preference = [x1, x2, x3]
        base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
        merged_model = efficient_merge_weights_with_preference(base_model_list, preference, temp_save_path)

        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        print(a, k, preference)
        all_rewards, all_full_prompts, all_full_responses = evaluate_model(merged_model, tokenizer, small_valid_dataset)
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_result = {}
        temp_list = []
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))
            temp_list.append(np.mean(evaluation_result['obtained_score{}'.format(i+1)]))
        result2.append(temp_list)
        if temp_list[1] > max_reward:
            max_reward = temp_list[1]
            k2 = k

    print('k1:', k1)
    print('k2:', k2)
    if process_id == 0:
        # save results to pickle
        import pickle
        with open(os.path.join(script_args.save_directory, script_args.wandb_name, 'result2.pkl'), 'wb') as f:
            pickle.dump(result2, f)



cnt4process= 0
to_merge_params_dict = None
for k in range(0, len(preferences)):
    preference = preferences[k]
    cnt4process+=1
    preference_org = preferences_org[k]

    x1 = preference[0]*solution1[0]*k1 + preference[1] * solution2[0]*k2
    x2 = preference[0]*solution1[1]*k1 + preference[1] * solution2[1]*k2
    x3 = (1-k1)*preference[0] + (1-k2)*preference[1]

    if process_id == 0:
        print('preference:', preference)
        print('x1, x2, x3:', x1, x2, x3)
        print('solution1:', solution1)
        print('solution2:', solution2)

    preference[0] = x1
    preference[1] = x2
    preference[2] = x3
    

    preference = torch.tensor(preference).cpu()
    temp_save_path = '/home/temp_models/{}'.format(script_args.wandb_name)
    # if process_id == 0:
    if len(preference) == 3:
        base_model_list = [base_model_name_1, base_model_name_2, base_model_name_3]
    else:
        base_model_list = [base_model_name_1, base_model_name_2]
    merged_model = efficient_merge_weights_with_preference(base_model_list, preference, temp_save_path)

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()
    # print(k, preference)
    all_rewards, all_full_prompts, all_full_responses = evaluate_model(merged_model, tokenizer, valid_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    if process_id == 0:
        evaluation_result = {
            'prompt': all_full_prompts,
            'response': all_full_responses,
        }
        for i in range(reward_models.num_rewards):
            evaluation_result['obtained_score{}'.format(i+1)] = all_rewards[i]
            print('total average obtained score {}: {}'.format(i+1, np.mean(evaluation_result['obtained_score{}'.format(i+1)])))

        dataframe = pd.DataFrame(evaluation_result)



        print(dataframe)
        try:
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(round(preference_org,1), round(1-preference_org,1))))
        except:
            # drop prompt and response
            dataframe = dataframe.drop(columns=['prompt', 'response'])
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'eval_data_pref{}_{}.csv'.format(round(preference_org,1), round(1-preference_org,1))))
