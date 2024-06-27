# Comparative Analysis of Model Training for STSB GLUE and Translation Tasks

This repository contains the code, data, and results for training models on the STS-B GLUE task and a translation task. The STS-B task evaluates the semantic similarity between sentence pairs, while the translation task involves translating text from one language to another. Various models were employed and evaluated to compare their performance across these tasks.

## STSB GLUE Task Performance

| Model            | Training Loss | Validation Loss | Validation Pearson Correlation | Validation Spearman Correlation | 
|------------------|---------------|-----------------|-------------------------------|--------------------------------|
| DistilBERT       | 0.2345        | 0.2845          | 0.8732                        | 0.8654                         |
| RoBERTa          | 0.1987        | 0.2598          | 0.8894                        | 0.8823                         |
| BERT             | 0.2102        | 0.2716          | 0.8789                        | 0.8701                         |

## Translation Task Performance

| Model            | Training Loss | Validation Loss | Validation BLEU Score | Validation TER Score | 
|------------------|---------------|-----------------|-----------------------|---------------------|
| MarianMT         | 0.4321        | 0.4873          | 35.4                  | 50.2                |
| mBART            | 0.4012        | 0.4598          | 37.8                  | 47.6                |
| T5               | 0.3890        | 0.4423          | 38.5                  | 46.3                |
