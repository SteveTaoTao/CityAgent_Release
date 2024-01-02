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

def prepare_texts(texts, conv_temp):
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
    q_and_a_body = {}
    device = 'cuda:{}'.format(args.gpu_id)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model = model.eval()
    image_path = "test_images/1.jpg"
    question = "trash"
    image = Image.open(image_path).convert("RGB")
    image = vis_processor(image).unsqueeze(0).to("cuda:0")
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    texts = prepare_texts([question], conv_temp)


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


if __name__ == "__main__":
    main()
