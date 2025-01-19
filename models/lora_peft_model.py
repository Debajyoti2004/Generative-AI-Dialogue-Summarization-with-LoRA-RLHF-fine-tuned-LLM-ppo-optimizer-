from peft import LoraConfig,PeftModel,PeftConfig, TaskType
from transformers import AutoModelForSeq2SeqLM
import torch

model_name = "google/flan-t5-base"
checkpoint_path = r"C:\Users\Debajyoti\OneDrive\Desktop\Generative_AI_(fine_tuned_model__detoxify_summarization\models\LoRA_fine_tuned_trained_flanT5\checkpoint-1"

lora_config = LoraConfig(
    r=32,
    lora_alpha =32,
    target_modules = ["q","v"],
    lora_dropout=0.05,
    bias = "none",
    task_type = TaskType.SEQ_2_SEQ_LM
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                           torch_dtype=torch.bfloat16)
peft_config = PeftConfig.from_pretrained(checkpoint_path)
peft_model = PeftModel.from_pretrained(
    model,
    checkpoint_path,
    lora_config = lora_config,
    torch_dtype = torch.bfloat16,
    device_map = "auto",
    is_trainable = True
)

print("PEFT model loaded successfully")