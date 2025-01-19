from transformers import AutoTokenizer, AutoModelForSequenceClassification, GenerationConfig
from tqdm import tqdm
import numpy as np

toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = AutoTokenizer.from_pretrained(toxicity_model_name)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(toxicity_model_name)

class EvaluateToxicity:
    def __init__(self, toxicity_classifier_model, main_model, toxicity_tokenizer, main_tokenizer, dataset, num_samples):
        self.toxicity_model = toxicity_classifier_model
        self.toxicity_tokenizer = toxicity_tokenizer
        self.tokenizer = main_tokenizer
        self.dataset = dataset
        self.num_samples = num_samples
        self.model = main_model

    def compute_toxicity(self, text, layer="softmax"):
        toxicity_ids = self.toxicity_tokenizer(text, return_tensors="pt").input_ids
        logits = self.toxicity_model(input_ids=toxicity_ids).logits
        probabilities = logits.softmax(dim=-1).tolist()[0]
        toxicity_score = probabilities[1]  # Assuming index 1 corresponds to toxicity
        return toxicity_score

    def evaluate_mean_and_variance(self):
        max_new_tokens = 100
        toxicities = []
        input_texts = []

        for i, sample in tqdm(enumerate(self.dataset)):
            input_text = sample['query']

            if i >= self.num_samples:
                break

            input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True).input_ids
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                top_k=0.0,
                top_p=1.0,
                do_sample=True
            )
            response_token_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config
            )
            generated_text = self.tokenizer.decode(response_token_ids[0], skip_special_tokens=True)

            toxicity_score = self.compute_toxicity(input_text + " " + generated_text)
            toxicities.append(toxicity_score)  # Use append instead of extend for scalar values

        mean = np.mean(toxicities)
        std = np.std(toxicities)

        return mean, std
