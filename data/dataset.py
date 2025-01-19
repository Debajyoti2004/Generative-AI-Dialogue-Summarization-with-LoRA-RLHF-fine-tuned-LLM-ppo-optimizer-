from datasets import load_dataset

dataset_name = "knkarthick/dialogsum"

original_dataset = load_dataset(dataset_name)
original_dataset.save_to_disk(r"C:\Users\Debajyoti\OneDrive\Desktop\Generative_AI_(fine_tuned_model__detoxify_summarization\data\dialogue_summarization_dataset")

print("Dataset saved at destined dir successfully!")
