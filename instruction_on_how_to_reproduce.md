### 1.Image size distributions:

    To obtain the distributions of the image size, you can run Code/image_ Size_ Distributions.py

### 2.Obejct detection results:

    To get the result image of MiniGPT-V2 detection, you can modify the image path and task, and then run Code/infer_detection.py to obtain it.

### 3.YOLO-World - Grounding DINO results:

    YOLO-World Detection results:
    https://github.com/AILab-CVC/YOLO-World
    To use YOLO-World for object detection, follow these steps: Install YOLO-World by following the instructions on the YOLO-World GitHub repository. Next, download the YOLO-World v2 L model weights. Then, run the demo script demo/gradio_demo.py. After launching the demo, upload the images you want to detect objects in and enter the relevant keywords. 

    GroundingDINO results:
    https://github.com/IDEA-Research/GroundingDINO
    According to the project's documentation, follow the installation steps and the 'inference_on_a_image.py' procedure to obtain the Grounding DINO detection results.

### 4.Fig. 13&14&15:

    Run the "interface_with_cityagent.py" to interact with cityagent to get results.
    1.Place the urban management-related regulatory documents in the data folder (we have already stored some). When the interactive program is running, the vector dataset will be automatically generated based on the documents in the folder.
    2.Input the path of the image you want to infer
    3.Input the question you want to ask, for example, "Is there trash in this picture?"
    4.Input the target you want to extract, for example, "trash."
    5.Specify what actions should be taken according to the regulations if trash is found.

### 5.MM-Vet evaluation

    See Code/README.md
