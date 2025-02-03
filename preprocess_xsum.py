from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


# Load the Xsum dataset and Xsum factual dataset
xsum_train = load_dataset("xsum", split="train")
xsum_val = load_dataset("xsum", split="validation")
xsum_test = load_dataset("xsum", split="test")
xsum_factual = load_dataset("google-research-datasets/xsum_factuality")['train']

# Combine Xsum splits into a single dataset
xsum_combined_dataset = concatenate_datasets([xsum_train, xsum_val, xsum_test])
print(len(xsum_combined_dataset))
print(type(xsum_combined_dataset[0]['id']))

# Convert the combined dataset into a dictionary for lookup by ID
xsum_combined = {entry['id']: entry for entry in xsum_combined_dataset}
print(len(xsum_combined))

# Create a list for final entries
final_entries = []

count1 = 0
# 1. Add all entries from XSum_Factuality
for example in xsum_factual:
    factual_id = str(example['bbcid'])  # Match ID format
    summary = example['summary']
    is_factual = example['is_factual']
    
    # Retrieve the document from XSum using the ID
    if factual_id in xsum_combined:
        count1 = count1+1
        doc = xsum_combined[factual_id]['document']
        final_entries.append({
            'id': factual_id,
            'doc': doc,
            'summary': summary,
            'is_factual': is_factual
        })

count2 = 0
# 2. Add gold summaries from XSum for IDs found in XSum_Factuality
for example in xsum_factual:
    factual_id = str(example['bbcid'])  # Match ID format
    
    # Retrieve the document and gold summary from XSum using the ID
    if factual_id in xsum_combined:
        count2 = count2+1
        doc = xsum_combined[factual_id]['document']
        gold_summary = xsum_combined[factual_id]['summary']
        final_entries.append({
            'id': factual_id,
            'doc': doc,
            'summary': gold_summary,
            'is_factual': 1  # Gold summaries are factual
        })

# Convert the final entries into a Hugging Face Dataset
final_dataset = Dataset.from_dict({
    'id': [entry['id'] for entry in final_entries],
    'doc': [entry['doc'] for entry in final_entries],
    'summary': [entry['summary'] for entry in final_entries],
    'is_factual': [entry['is_factual'] for entry in final_entries]
})

print(count1, count2)

# Optional: Save the dataset to disk
final_dataset.save_to_disk("xsum_factual_combined_with_gold")