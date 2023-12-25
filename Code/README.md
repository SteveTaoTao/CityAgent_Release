# CityAgent: a Knowledge-Enhanced Multimodal Framework for Advanced Urban Management

## Introduction
CityAgent is a knowledge-enhanced multimodal framework based on Large Multimodal Model (LMM) and Retrieval Augmented Generation (RAG), aiming to provide further support for urban management.

## Content

1.  [visible_grounding.py: Visualize training data](DB_tools/visible_tools/visible_grounding.py)                        
2.  [make_crop_data_work.py: Crop images](DB_tools/make_db/make_crop_data_work.py)
3.  [infer_all_mmvet.py: Evaluation](infer_all_mmvet.py)
4.  [rag_infer.py: Inference code for RAG module](RagCode/lin_rag.py)
5.  [fine-tune weight](https://figshare.com/s/fddd31a9906038bda8e0)
6.  [vqa_cli_infer.py: Inference code for VQA ](vqa_cli_infer.py)

## Installation

1.  MiniGPT-v2  [GPTv2_README](README_MINI_BASE.md)


## Quick Start
### Environment Setup：
Follow [GPTv2_README](README_MINI_BASE.md)  to install the necessary environment. Ensure your system meets all prerequisites, including the installation of all required dependencies and libraries.

### (Optional): Data Preprocessing
If you wish to fine-tune the model with your own data, please use the [make_crop_data_work.py: Crop images](DB_tools/make_db/make_crop_data_work.py) script. This script will help you convert your target detection dataset into the format required by MiniGPTv2.

### (Optional): Data Visualization
To verify the correctness of the data conversion, you can use the [visible_grounding.py: Visualize training data](DB_tools/visible_tools/visible_grounding.py) code for data visualization. This step will help you confirm whether the data format conversion results meet expectations.

### (Optional): Model Fine-Tuning and Inference
Using Fine-Tuned Weights: If you have already fine-tuned the model, you can configure the fine-tuned weights in mmvet_infer.py for performance evaluation.
Using  [fine-tune weight](https://figshare.com/s/fddd31a9906038bda8e0): Alternatively, you can download our fine-tune weights and use the  [vqa_cli_infer.py: Inference code for VQA ](vqa_cli_infer.py) code to perform direct model inference.

