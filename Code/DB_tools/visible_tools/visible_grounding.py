import cv2
import json
import re
import numpy as np


json_file_path = "/media/DB/Filteredflickr-30k/phrasetobbox.json"
image_base_path = "/media/DB/Filteredflickr-30k/flickr30k_images/"
all_get = json.loads(open(json_file_path,encoding="utf-8").read())
for every_obj in all_get:
    get_image_path = image_base_path+every_obj["image_id"]+".jpg"
    get_image_box = every_obj["bbox"]

    print(every_obj["phrase"])
    bbox_list = get_image_box.split('<delim>')
    get_image_src =  cv2.imdecode(np.fromfile(get_image_path, dtype=np.uint8), -1)
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
    get_image_src = cv2.resize(get_image_src,(1000,600))
    cv2.imshow("dd",get_image_src)
    cv2.waitKey(0) 
