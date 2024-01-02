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
7.  [infer_detection.py: Inference code for get detection result](infer_detection.py)
8.  [interface_with_cityagent.py: Inference code for interface with cityagent](interface_with_cityagent.py)
9.  [Our mock data](https://figshare.com/s/022d60d9e3cc3759cf64 )

## Installation
1.  Git clone [MiniGPT-v2 Project](https://github.com/Vision-CAIR/MiniGPT-4)
2.  Follow the installation steps in [MiniGPT-v2_README](README_MINI_BASE.md)
3.  Move all our project files to the root directory of MiniGPT-v2 Project and follow Quick Start to execute.
4. (Optional):If you want to use our [MM-Vet evaluation code](infer_all_mmvet.py), you can obtain the [MM-Vet data](https://huggingface.co/datasets/Otter-AI/MMVet/tree/main) and then place the folder in the root directory of this project.

## Quick Start
### Environment Setup：
Follow [GPTv2_README](README_MINI_BASE.md)  to install the necessary environment. Ensure your system meets all prerequisites, including the installation of all required dependencies and libraries.

### (Optional): Data Preprocessing
If you wish to fine-tune the model with your own data, please use the [make_crop_data_work.py: Crop images](DB_tools/make_db/make_crop_data_work.py) script. This script will help you convert your target detection dataset into the format required by MiniGPT-v2. You can download [Our mock data](https://figshare.com/s/022d60d9e3cc3759cf64 ) and then use  script to convert the data into a format that MiniGPT-v2 can train.

### (Optional): Data Visualization
To verify the correctness of the data conversion, you can use the [visible_grounding.py: Visualize training data](DB_tools/visible_tools/visible_grounding.py) code for data visualization. This step will help you confirm whether the data format conversion results meet expectations.

### (Optional): Model Fine-Tuning and Inference
Using Fine-Tuned Weights: If you have already fine-tuned the model, you can configure the fine-tuned weights in mmvet_infer.py for performance evaluation.
Using  [fine-tune weight](https://figshare.com/s/fddd31a9906038bda8e0): Alternatively, you can download our fine-tune weights and use the  [vqa_cli_infer.py: Inference code for VQA ](vqa_cli_infer.py) code to perform direct model inference. If you want to get the result image of MiniGPT-V2 detection, you can modify the image path and task, and then run [infer_detection.py](infer_detection.py) to obtain it.


### (Optional): Interface with cityagent
When you have completed the environment setup and obtained the weights, you can run the [interface_with_cityagent.py](interface_with_cityagent.py) to interact with cityagent.

    1.Place the urban management-related regulatory documents in the data folder (we have already stored some). When the interactive program is running, the vector dataset will be automatically generated based on the documents in the folder.
    2.Input the path of the image you want to infer
    3.Input the question you want to ask, for example, "Is there trash in this picture?"
    4.Input the target you want to extract, for example, "trash."
    5.Specify what actions should be taken according to the regulations if trash is found.
