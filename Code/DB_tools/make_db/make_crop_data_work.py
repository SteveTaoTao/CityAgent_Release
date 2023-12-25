import xml.etree.ElementTree as ET
from PIL import Image
import os

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def update_xml(xml_path, crop_area, output_xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    size.find('width').text = str(crop_area[2] - crop_area[0])
    size.find('height').text = str(crop_area[3] - crop_area[1])

    object_count = 0
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)

        if x1 >= crop_area[0] and y1 >= crop_area[1] and x2 <= crop_area[2] and y2 <= crop_area[3]:
            bbox.find('xmin').text = str(x1 - crop_area[0])
            bbox.find('ymin').text = str(y1 - crop_area[1])
            bbox.find('xmax').text = str(x2 - crop_area[0])
            bbox.find('ymax').text = str(y2 - crop_area[1])
            object_count += 1
        else:
            root.remove(obj)

    if object_count >= 2:
        tree.write(output_xml_path)
        return True
    else:
        return False

def is_target_fully_included(target_box, crop_area):
    return target_box[0] >= crop_area[0] and target_box[1] >= crop_area[1] and target_box[2] <= crop_area[2] and target_box[3] <= crop_area[3]

def generate_crop_areas(width, height, target_size, target_boxes):
    crop_areas = []
    for y in range(0, height - target_size[1] + 1, target_size[1]):
        for x in range(0, width - target_size[0] + 1, target_size[0]):
            new_area = (x, y, x + target_size[0], y + target_size[1])

            included_targets = [target for target in target_boxes if is_target_fully_included(target, new_area)]
            if len(included_targets) < 2:
                continue

            if not any(calculate_iou(new_area, area) > 0.2 for area in crop_areas):
                crop_areas.append(new_area)

    return crop_areas

def crop_images(image_folder, xml_folder, output_folder, target_size=(616, 616), iou_threshold=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            xml_path = os.path.join(xml_folder, image_file.split('.')[0] + '.xml')

            image = Image.open(image_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width, height = image.size

            target_boxes = []
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)
                target_boxes.append((x1, y1, x2, y2))

            crop_areas = generate_crop_areas(width, height, target_size, target_boxes)

            for i, crop_area in enumerate(crop_areas):
                cropped_image = image.crop(crop_area)
                cropped_image_path = os.path.join(output_folder,"images", f"{image_file.split('.')[0]}_cropped_{i}.jpg")
                output_xml_path = os.path.join(output_folder,"xml" ,f"{image_file.split('.')[0]}_cropped_{i}.xml")

                if update_xml(xml_path, crop_area, output_xml_path):
                    cropped_image.save(cropped_image_path,format='JPEG')

image_path = "DB/base_data/images/"
xml_path = "DB/base_data/xml/"
crop_images(image_path, xml_path, 'DB/crop_data/')
