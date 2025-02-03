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

## Finetuning LLAMA 2 7b

## Collecting activations
