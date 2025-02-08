# Factual-LLM
Making LLM generated summaries more factual by steering generations. This work includes the following:
1. Preparing dataset(BBC XSum and Summeval)
2. Finetuning LLAMA 2 7b for classification task
3. Collecting activations from heads and layers
4. Steering LLM generations for more factual summaries

## Data preparation
For this project we are working with [BBC XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)  and [Summeval](https://huggingface.co/datasets/davidanugraha/SummEval) datasets. We preprocess both these datasets for our tasks. We use XSum for finetuning LLAMA 7b and we use Summeval to collect the activations.
### XSum
We combine XSum and [XSum factuality](https://huggingface.co/datasets/google-research-datasets/xsum_factuality) to create a dataset containing both machine generated and gold summaries of documents along with their id and labels. The label is defined with two classes 1 for gold/factual summaries and 0 for non-factual summaries. Therefore, the new combined dataset has `11,194` entries. We use this dataset to finetune llama 2 7b model with LoRA with 4:1 training-validation ratio. 

### Summeval
For Summeval, we use the 1700 entries and update the label columns such that expert_consistency is considered as the `is_factual` label similar to XSum. The value of the label is also updated to 0 and 1 and we do this by creating a threshold value of 3.5. If the original value of expert_consistency is below 3.5 then the label is updated to 0(non-factual) and if it is above, it is updated 1(factual).

* To create the datasets, simply run the preprocess_<dataset_name>.py/ipynb files and the datasets will be created locally.
* To load this dataset, use the same format as Huggingface dataset.
  ```
  dataset = Dataset.load_from_disk("path/to/your/dataset")
  print(dataset)
  ```

## Finetuning LLAMA 2 7b
We use the LLAMA 2 7b hf model for finetuning. We configure the LoRA parameters as follows: r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="SEQ_CLS". Our finetuning results are given as below:

| Epoch | Training Loss | Validation Loss |    Accuracy |      F1|
--------|---------------|-----------------|-------------|--------|
   1    | 0.691200      | 0.684779        | 0.542282    |0.351610|
   2    | 0.495100      | 0.421361        | 0.751230    |0.750741|
   3    | 0.367600      | 0.383919        | 0.789418    |0.766380|

* To finetune the model from scratch, simply run the `finetune_llama.ipynb`. 
* To load the finetuned model directly without finetuning, use the code below:
  
  ```
  model = LlamaForSequenceClassification.from_pretrained("trinadutta/finetuned_llama")
  tokenizer = LlamaTokenizer.from_pretrained("trinadutta/finetuned_llama")
  ```
## Collecting activations
We collect activations from the finetuned llama model on the preproceesed Summeval dataset. For this step, we use [Pyvene](https://stanfordnlp.github.io/pyvene/tutorials/pyvene_101.html) wrappers to collect layerwise and headwise activations.
