from transformers import LlamaForSequenceClassification, LlamaTokenizer
from datasets import Dataset
import torch
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
from tqdm import tqdm
from utils import get_llama_activations_pyvene
import numpy as np

model = LlamaForSequenceClassification.from_pretrained("trinadutta/finetuned_llama")
tokenizer = LlamaTokenizer.from_pretrained("trinadutta/finetuned_llama")

dataset = Dataset.load_from_disk("/home/trina/NLP/REpyvene/fix_hallucination/cnndm_factual/train")
print(dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)

def formatter(dataset, tokenizer):
    prompts = []
    labels = []
    for i, example in enumerate(dataset):
        prompt = f"{example['doc']} [SEP] {example['summary']}"
        # Tokenize the prompt
        tokenized = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt"
        )
        
        # Extract input IDs and ensure they are squeezed for compatibility
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Move the tokenized data to the desired device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        prompts.append(input_ids)
        labels.append(example["is_factual"])
    return prompts, labels


prompts, labels = formatter(dataset, tokenizer)


model_name = "llama_xsum"

collectors = []
pv_config = []
for layer in range(model.config.num_hidden_layers):
    collector = Collector(multiplier=0, head=-1)  # Collect all head activations
    collectors.append(collector)
    pv_config.append({
        "component": f"model.layers[{layer}].self_attn.o_proj.input",
        "intervention": wrapper(collector),
    })
    

collected_model = pv.IntervenableModel(pv_config, model)
all_layer_wise_activations = []
all_head_wise_activations = []


for prompt in tqdm(prompts):  # Prompts should be tokenized input_ids

    # Get activations from the model
    layer_wise_activations, head_wise_activations, _ = get_llama_activations_pyvene(collected_model, collectors, prompt, device)
    
    # Store layer-wise activations (taking only the last token's representation)
    all_layer_wise_activations.append(layer_wise_activations[:, -1, :].copy())  # Shape: (num_layers, num_features)

    # Store head-wise activations (ensuring proper shape)
    all_head_wise_activations.append(head_wise_activations.copy())  # Shape: (num_layers, num_heads, num_features)

# Convert to numpy arrays
all_layer_wise_activations = np.array(all_layer_wise_activations)  # Shape: (num_samples, num_layers, num_features)
all_head_wise_activations = np.array(all_head_wise_activations)  # Shape: (num_samples, num_layers, num_heads, num_features)



np.save(f'/home/trina/NLP/REpyvene/fix_hallucination/new_features/{model_name}_layer_wise.npy', all_layer_wise_activations)
np.save(f'/home/trina/NLP/REpyvene/fix_hallucination/new_features/{model_name}_head_wise.npy', all_head_wise_activations)
np.save(f'/home/trina/NLP/REpyvene/fix_hallucination/new_features/{model_name}_labels.npy', labels)
