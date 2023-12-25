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
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts


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
    image_path = "./mm-vet/images/v1_8.png"
    question = "What will the girl on the right write on the board?"
    cfg = Config(args)
    device = 'cuda:{}'.format(args.gpu_id)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    model = model.eval()
    image = Image.open(image_path).convert("RGB")
    image = vis_processor(image).unsqueeze(0).to("cuda:0")
    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""
    texts = prepare_texts([question], conv_temp)
    print("texts:",texts)
    answers = model.generate(image, texts, max_new_tokens=1024, do_sample=False)
    print(f"Answer: {answers}")


if __name__ == "__main__":
    main()
