from datasets import load_from_disk
from transformers import AutoTokenizer

class Preprocessing_dataset:
    def __init__(self, model_name, dataset_path, input_min_text_length, input_max_text_length, type=None):
        self.dataset = load_from_disk(dataset_path)[type]
        self.filtered_dataset = self.dataset.filter(
            lambda x: len(x['dialogue']) > input_min_text_length and len(x['dialogue']) < input_max_text_length,
            batched=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_prompt(self, sample):
        prompt = f"""
        Summarize the following conversation:
        
        {sample['dialogue']}
        
        Summary:
        """
        return prompt.strip()

    def tokenize(self, sample):
        prompt = self.create_prompt(sample)
        sample['input_ids'] = self.tokenizer.encode(prompt, truncation=True, max_length=512)
        sample["query"] = self.tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        return sample

    def update_dataset(self):
        dataset = self.filtered_dataset.map(self.tokenize, batched=False)
        dataset.set_format(type="torch")
        dataset_splits = dataset.train_test_split(
            test_size=0.2,
            shuffle=True,
            seed=42
        )
        return dataset_splits
