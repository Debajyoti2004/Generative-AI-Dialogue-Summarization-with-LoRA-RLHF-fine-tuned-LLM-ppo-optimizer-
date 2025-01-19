from trl import AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from lora_peft_model import peft_model
import torch

ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(peft_model,
                                                               torch_dtype = torch.bfloat16,
                                                               is_trainable =True)

ref_model = create_reference_model(ppo_model)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"\ntrainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


print(f"trainable ppo model weights:{print_number_of_trainable_model_parameters(ppo_model)}")
print(f"freeze weight reference model :{print_number_of_trainable_model_parameters(ref_model)}")

print("ppo_model loaded successfully!")
print("refertence freeze weight model loaded successfully!")