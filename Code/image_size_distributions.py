import os
import cv2

all_dirs = ["AOK-VQA", "Filteredflickr-30k", "LLAVA", "OCR-VQA", "RefCOCO", "RefCOCOg", "VisualGenome", "coco_captions","COCOVQA", "FilteredUnnaturalInstruction", "GQA", "Multi-taskConversation", "OKVQA", "RefCOCO+", "TextCaps", "coco"]

def get_image_size_distribution(base_image_folder, sub_folders):
    size_distribution = {
        '0-500px': 0,
        '500-1000px': 0,
        '1000-1500px': 0,
        '1500-2000px': 0,
        '2000-2500px': 0,
        '2500-3000px': 0,
        'bigger than 3000px': 0
    }
    total_images = 0

    for sub_folder in sub_folders:
        folder_path = os.path.join(base_image_folder, sub_folder)
        if not os.path.exists(folder_path):
            print(f"dir {folder_path} doesn't exist.")
            continue
        print("folder_path:",folder_path)
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_path = os.path.join(dirpath, filename)
                    img = cv2.imread(image_path)
                    if img is not None:
                        total_images += 1
                        height, width = img.shape[:2]
                        max_side = max(width, height)

                        if max_side <= 500:
                            size_distribution['0-500px'] += 1
                        elif max_side <= 1000:
                            size_distribution['500-1000px'] += 1
                        elif max_side <= 1500:
                            size_distribution['1000-1500px'] += 1
                        elif max_side <= 2000:
                            size_distribution['1500-2000px'] += 1
                        elif max_side <= 2500:
                            size_distribution['2000-2500px'] += 1
                        elif max_side <= 3000:
                            size_distribution['2500-3000px'] += 1
                        else:
                            size_distribution['bigger than 3000px'] += 1

    for key in size_distribution:
        size_distribution[key] = (size_distribution[key] / total_images) * 100 if total_images > 0 else 0

    return size_distribution

base_image_folder = '/media/train_data/'
distribution = get_image_size_distribution(base_image_folder, all_dirs)
print(distribution)
