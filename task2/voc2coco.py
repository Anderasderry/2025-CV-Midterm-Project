import os
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import pycocotools.mask as mask_utils
from tqdm import tqdm

IOU_THRESHOLD = 0.2
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def convert_voc_to_coco(voc_root, output_root):
    splits = {'train': 'train2017', 'val': 'val2017'}
    for split, coco_split in splits.items():
        split_file = os.path.join(voc_root, 'ImageSets/Segmentation', f'{split}.txt')
        if not os.path.exists(split_file):
            print(f"警告：未找到 {split_file}，跳过")
            continue

        with open(split_file) as f:
            file_names = [line.strip() for line in f.readlines()]

        coco_data = {
            "info": {"description": f"VOC2012-COCO {split}"},
            "licenses": [],
            "categories": [{"id": i + 1, "name": cls} for i, cls in enumerate(VOC_CLASSES)],
            "images": [],
            "annotations": []
        }

        annotation_id = 1
        unmatched_count = 0
        error_count = 0
        img_output_dir = os.path.join(output_root, coco_split, 'VOC2012', 'JPEGImages')
        os.makedirs(img_output_dir, exist_ok=True)

        for img_id, file_name in enumerate(tqdm(file_names, desc=f'处理 {split}')):
            img_path = os.path.join(voc_root, 'JPEGImages', f'{file_name}.jpg')
            xml_path = os.path.join(voc_root, 'Annotations', f'{file_name}.xml')
            seg_path = os.path.join(voc_root, 'SegmentationObject', f'{file_name}.png')

            if not os.path.exists(img_path) or not os.path.exists(xml_path) or not os.path.exists(seg_path):
                error_count += 1
                continue

            with Image.open(img_path) as img:
                width, height = img.size

            # 拷贝图片
            shutil.copy(img_path, os.path.join(img_output_dir, f'{file_name}.jpg'))

            coco_data["images"].append({
                "id": img_id,
                "file_name": f'VOC2012/JPEGImages/{file_name}.jpg',
                "width": width,
                "height": height
            })

            tree = ET.parse(xml_path)
            root = tree.getroot()

            valid_objects = []
            for obj in root.findall('object'):
                if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
                    continue
                cls_name = obj.find('name').text.strip()
                if cls_name in VOC_CLASSES:
                    valid_objects.append({
                        "class": cls_name,
                        "bndbox": obj.find('bndbox')
                    })

            seg_mask = np.array(Image.open(seg_path))

            for obj in valid_objects:
                try:
                    xmin = int(float(obj['bndbox'].find('xmin').text))
                    ymin = int(float(obj['bndbox'].find('ymin').text))
                    xmax = int(float(obj['bndbox'].find('xmax').text))
                    ymax = int(float(obj['bndbox'].find('ymax').text))
                except:
                    continue

                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(width - 1, xmax)
                ymax = min(height - 1, ymax)
                if xmax <= xmin or ymax <= ymin:
                    continue

                bbox_mask = np.zeros_like(seg_mask, dtype=np.uint8)
                bbox_mask[ymin:ymax+1, xmin:xmax+1] = 1

                instance_ids = np.unique(seg_mask)
                instance_ids = instance_ids[instance_ids != 0]

                best_iou = 0
                best_id = -1

                for inst_id in instance_ids:
                    inst_mask = (seg_mask == inst_id).astype(np.uint8)
                    iou = calculate_iou(inst_mask, bbox_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_id = inst_id

                if best_iou < IOU_THRESHOLD:
                    unmatched_count += 1
                    continue

                binary_mask = (seg_mask == best_id).astype(np.uint8)

                try:
                    rle = mask_utils.encode(np.asfortranarray(binary_mask))
                    rle['counts'] = rle['counts'].decode('ascii')
                except Exception:
                    continue

                coco_bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]

                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": VOC_CLASSES.index(obj['class']) + 1,
                    "segmentation": rle,
                    "area": float(binary_mask.sum()),
                    "bbox": coco_bbox,
                    "iscrowd": 0
                })
                annotation_id += 1

        anno_out_path = os.path.join(output_root, 'Annotations', f'instances_{coco_split}.json')
        os.makedirs(os.path.dirname(anno_out_path), exist_ok=True)
        with open(anno_out_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"\n 已保存 {split} 到 {anno_out_path}")
        print(f"错误图像数：{error_count}，未匹配实例数：{unmatched_count}")

if __name__ == '__main__':
    VOC_ROOT = 'VOCdevkit/VOC2012'
    OUTPUT_DIR = 'data/coco'
    convert_voc_to_coco(VOC_ROOT, OUTPUT_DIR)