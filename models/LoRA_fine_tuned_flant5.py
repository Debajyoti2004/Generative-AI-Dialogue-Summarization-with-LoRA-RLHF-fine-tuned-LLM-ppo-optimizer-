from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch
from datasets import load_from_disk

tokenized_datasets = load_from_disk(r'C:\Users\Debajyoti\OneDrive\Desktop\Generative_AI with flan-t5 (fine tuned)\tokenized_datasets')

model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype = torch.float16)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q","v"],
    lora_dropout = 0.5,
    bias= "none",
    task_type = TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(
    original_model,
    lora_config
)
def print_number_of_trainable_model_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    return trainable_params

trainable_parameters = print_number_of_trainable_model_parameters(peft_model)


output_dir = r"C:\Users\Debajyoti\OneDrive\Desktop\Generative_AI with flan-t5 (fine tuned)\models\LoRA_fine_tuned_trained_flanT5"

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    per_device_train_batch_size=2, 
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)
peft_trainer.train()
print("Trained successfully completed!")
