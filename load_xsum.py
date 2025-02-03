from datasets import Dataset

def load_xsum():
    # Load the dataset from the local disk path
    dataset = Dataset.load_from_disk("xsum_factual_combined_with_gold")
    return dataset
