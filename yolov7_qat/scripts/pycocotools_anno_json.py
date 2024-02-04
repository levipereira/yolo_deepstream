import argparse
import os
import yaml
import json
import cv2
from tqdm import tqdm

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_min = max(0, (x_center - width / 2) * img_width)
    y_min = max(0, (y_center - height / 2) * img_height)
    x_max = min(img_width, (x_center + width / 2) * img_width)
    y_max = min(img_height, (y_center + height / 2) * img_height)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def build_pycocotools_anno_json(data_yaml, output_file_json):
    # Load YAML file
    with open(data_yaml, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader) 

    # Create categories based on the data
    categories = [{"id": i + 1, "name": name, "supercategory": name} for i, name in enumerate(data['names'])]

    # Create instances JSON data
    instances_data = {
        "info": {
        "description": "Custom Dataset",
        "version": "1.0",
        "year": 2024,
        "contributor": "",
        "date_created": "2024/01/01"
        },
        "licenses": [
            {
                "url": "http://dummy",
                "id": 1,
                "name": "dummy"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = ""
    annotation_id = 1
    processed_files = set()

    # Read paths of images from the specified text file
    with open(data['val'], 'r') as image_list_file:
        image_paths = image_list_file.read().splitlines()

    # Count total images for tqdm progress bar
    total_images = len(image_paths)

    with tqdm(total=total_images, desc="Processing Images") as pbar:
        for image_path in image_paths:
            if os.path.exists(image_path):
                _, extension = os.path.splitext(image_path)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                if image_name not in processed_files:
                    # Add the image to the processed files set to ensure uniqueness
                    processed_files.add(image_name)
                    if image_name.isdigit():
                        image_id=str(int(image_name))
                    else:
                        image_id=image_name
                    # Read image size using OpenCV
                    img = cv2.imread(image_path)
                    if img is not None:
                        img_height, img_width, _ = img.shape

                        # Add image information
                        image_info = {"license": 1, 
                                    "file_name": image_name + extension, 
                                    "height": img_height,
                                    "width": img_width,
                                    "id": image_id}
                        instances_data["images"].append(image_info)

                        # Read annotation file if it exists
                        annotation_path = os.path.splitext(image_path)[0] + '.txt'
                        if os.path.exists(annotation_path):  # Check if file exists
                            with open(annotation_path, 'r') as ann_file:
                                for line in ann_file:
                                    label = line.strip().split()
                                    category_id = int(label[0]) + 1  # Assuming YOLO format starts from 0
                                    yolo_bbox = list(map(float, label[1:]))
                                    coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)

                                    # Add annotation information
                                    annotation_info = {
                                        "id": annotation_id,
                                        "image_id": image_id,
                                        "category_id": category_id,
                                        "bbox": coco_bbox,
                                        "area": coco_bbox[2] * coco_bbox[3],
                                        "segmentation": [],
                                        "iscrowd": 0  # Assuming no crowd instances
                                    }
                                    instances_data["annotations"].append(annotation_info)
                                    annotation_id += 1
                        else:
                            print(f"Error: Annotation file not found for image {image_name}. pycocotools will not be used.")
                            return False
                    else:
                        print(f"Error: Failed to read image {image_path}. pycocotools will not be used.")
                        return False
                else:
                    print(f"Error: Image filename {image_name} is not unique to dataset. pycocotools will not be used. ")
                    return False
                pbar.update(1)  # Update progress bar
            else:
                print(f"Error: Image {image_name} not found. pycocotools will not be used.")
                return False
    # Save instances JSON data to file
    with open(output_file_json, 'w') as f:
        json.dump(instances_data, f)
    return True

def main():
    parser = argparse.ArgumentParser(description='Create instances JSON file from dataset YAML  Validating only Dataset')
    parser.add_argument('--data', type=str, help='Path to dataset YAML file', required=True)
    parser.add_argument('--output', type=str, help='Path to output JSON file', required=True)
    args = parser.parse_args()

    build_pycocotools_anno_json(args.data, args.output)

if __name__ == "__main__":
    main()
