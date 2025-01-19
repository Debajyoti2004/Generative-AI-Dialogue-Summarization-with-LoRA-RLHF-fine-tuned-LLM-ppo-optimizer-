from trl import PPOConfig, PPOTrainer
from transformers import AutoTokenizer
from datasets import load_from_disk
import torch
from tqdm import tqdm
from create_ppo_and_ref_model import ppo_model, ref_model
from trl.core import LengthSampler  
# Set parameters
learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16
model_name = "google/flan-t5-base"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load dataset
dataset_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Generative_AI_(fine_tuned_model__detoxify_summarization\data\preprocess_datasets"
dataset = load_from_disk(dataset_path)

# Define the collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# Set up PPOConfig
config = PPOConfig(  
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

# Set up PPOTrainer
ppo_trainer = PPOTrainer(
    config=config, 
    model=ppo_model, 
    ref_model=ref_model, 
    tokenizer=tokenizer, 
    dataset=dataset["train"], 
    data_collator=collator
)

# Fine-tuning function
def fine_tuning_instruct_model(output_min_length, output_max_length, ppo_trainer, evaluate_toxicity, max_ppo_epoch=10):

    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    generation_kwargs = {
        "min_length": 5,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True
    }
    
    for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        if step >= max_ppo_epoch:
            break

        prompt_tensors = batch["input_ids"]
        summary_tensors = []

        # Generate summaries for each prompt in the batch
        for prompt_tensor in prompt_tensors:
            max_new_token = output_length_sampler()
            generation_kwargs["max_new_tokens"] = max_new_token
            summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)
            summary_tensors.append(summary.squeeze()[-max_new_token:])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in summary_tensors]

        query_response_pairs = [q + r for q, r in zip(batch["query"], batch["response"])]
        
        rewards_tensors = []
        for query_and_response in query_response_pairs:
            reward = evaluate_toxicity.compute_toxicity(query_and_response)
            reward_tensor = torch.tensor(reward)
            rewards_tensors.append(reward_tensor)

        # PPO step
        stats = ppo_trainer.step(prompt_tensors, summary_tensors, rewards_tensors)
        ppo_trainer.log_stats(stats, batch, rewards_tensors)

        # Log PPO stats
        print(f'objective/kl: {stats["objective/kl"]}')
        print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
        print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
        print('-' * 100)

    return ppo_trainer, ppo_model


