import argparse
import os
import random
from collections import defaultdict

import cv2
import re
from PIL import Image, ImageDraw, ImageFont
font = ImageFont.truetype('simhei.ttf', 15, encoding="utf-8")

import numpy as np
import torch
import html
import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
 import os
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory


os.environ["OPENAI_API_KEY"] = "valid openai api key"

from langchain.document_loaders import DirectoryLoader



import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


CONV_VISION_minigptv2 = Conversation(
    system="",
    roles=("<s>[INST] ", " [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

def prepare_texts_vqa(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> [vqa]{}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def prepare_texts_det(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> [detection]{}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

# load metadata
# Download mm-vet.zip and `unzip mm-vet.zip` and change the path below
use_sub_set = False
decimal_places = 1 # number of decimal places to round to



def main():
    parser = argparse.ArgumentParser(description="CLI for Image and Question Inference")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml', help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    cfg = Config(args)




   
    loader = DirectoryLoader('RagCode/regulation_data', glob="**/[!.]*",show_progress=True)
        # Set glob="**/[!.]*" to load all files except hidden ones
        # Set show_progress=True to display loading progress
    docs = loader.load()
    if len(docs):
        print("Files loaded successfully")


    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\r\n", "\n\n", "\n\n\u3000", "\n\u3000\u3000"], # customized separators
        chunk_size = 400,
        chunk_overlap = 200
    )

    docs_splits = text_splitter.split_documents(docs)
    print("len(docs_splits): ",len(docs_splits)) 



    vectorstore = Chroma(
        collection_name="regulations",
        embedding_function=OpenAIEmbeddings() #ada by default
    )

    #### Document Database Settings

    store = InMemoryStore() # Store documents in memory
    id_key = "doc_id" # Document id key


    #### Set Retriever, Specify Document Database and Vector Database
    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, 
        docstore=store, 
        id_key=id_key,
    )

    #### Document Data Processing

    doc_ids = [str(uuid.uuid4()) for _ in docs_splits]

    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 150, 
        chunk_overlap = 50, 
        )

    sub_docs_splits = []

    for i, doc in enumerate(docs_splits):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc]) 
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id 
        sub_docs_splits.extend(_sub_docs)



    retriever.vectorstore.add_documents(sub_docs_splits)
    retriever.docstore.mset(list(zip(doc_ids, docs_splits)))

    #### similarity search
    similar_docs = retriever.vectorstore.similarity_search("your questions")
    relevant_docs = retriever.get_relevant_documents("your questions")



    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(model_name='gpt-3.5-turbo', temperature=0), retriever, memory=memory)
    




    device = 'cuda:{}'.format(args.gpu_id)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model = model.eval()
    image_path = input("Please enter the file name: ")
    question = "trash"
    image = Image.open(image_path).convert("RGB")
    image = vis_processor(image).unsqueeze(0).to("cuda:0")
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    questionvqa = input("Please enter your question: ")
    texts = prepare_texts_vqa([questionvqa], conv_temp)


    answers = model.generate(image, texts, max_new_tokens=1024, do_sample=False)
    answer_txt = answers[0]
    print(answer_txt)

    questiondet = input("Please enter your question: ")
    texts = prepare_texts_det([questionvqa], conv_temp)


    answers = model.generate(image, texts, max_new_tokens=1024, do_sample=False)
    answer_txt = answers[0]
    
    get_image_boxes = answer_txt.split(" ")[-1]
    bbox_list = get_image_boxes.split('<delim>')
    get_image_src =  cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    image_h,image_w,_ = get_image_src.shape
    for bbox_string in bbox_list:
        numbers = re.findall(r'\d+', bbox_string)
        # Converting the extracted strings to integers
        integers = [int(num) for num in numbers]
        if len(integers) == 4:
            x0, y0, x1, y1 = int(integers[0]), int(integers[1]), int(integers[2]), int(integers[3])
            new_x0  =int(  x0/100 * image_w)
            new_y0  =int(  y0/100 * image_h)
            new_x1  =int(  x1/100 * image_w)
            new_y1  =int(  y1/100 * image_h)
            get_image_src = cv2.rectangle(get_image_src, (new_x0, new_y0), (new_x1, new_y1), (0,0,255), 3)
    cv2.imwrite("result.jpg", get_image_src)
    print("model: Result save as name result.jpg")

    questionRAG = input("Please enter your question: ")
    result = qa({"question": questionRAG})
    print("model: ",result)
if __name__ == "__main__":
    main()
