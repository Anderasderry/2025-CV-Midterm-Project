import os
import random
import traceback
import torch
import mmcv
import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmcv.transforms import Compose

# 路径配置
config_file = 'work_dirs/mask-rcnn_r50_fpn_1x_coco/mask-rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/mask-rcnn_r50_fpn_1x_coco/epoch_30.pth'
base_out_dir = 'visualizations'
input_dir = os.path.join(base_out_dir, 'in')
proposals_dir = os.path.join(base_out_dir, 'out', 'proposals')
predictions_dir = os.path.join(base_out_dir, 'out', 'predictions')
backend_args = None

# 创建输出目录
os.makedirs(input_dir, exist_ok=True)
os.makedirs(proposals_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# 初始化模型
try:
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
except Exception as e:
    print(f"[ERROR] 初始化模型失败：{e}")
    traceback.print_exc()
    exit(1)

# 初始化可视化器
visualizer_cfg = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend', save_dir=base_out_dir),
        dict(type='TensorboardVisBackend', save_dir=os.path.join(base_out_dir, 'tensorboard'))
    ],
    name='visualizer'
)
visualizer = VISUALIZERS.build(visualizer_cfg)
visualizer.dataset_meta = {
    'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person',
                'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255),
                (190, 153, 153), (180, 165, 180), (90, 86, 231), (210, 120, 180), (102, 121, 66),
                (0, 255, 0), (0, 0, 142), (0, 60, 100), (0, 0, 230), (0, 80, 100),
                (46, 191, 191), (81, 0, 21), (220, 20, 60), (255, 245, 0), (139, 0, 0)]
}

# 定义图像预处理管道
test_pipeline = Compose([
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
])

def preprocess_image(img_path):
    try:
        data = dict(
            img_path=img_path,
            img_id=os.path.splitext(os.path.basename(img_path))[0]
        )
        data = test_pipeline(data)
        img_tensor = data['inputs'].to('cuda:0')
        img_tensor = img_tensor.float()
        data_sample = data['data_samples']
        img = mmcv.imread(img_path)
        return img_tensor, data_sample, img
    except Exception as e:
        print(f"[ERROR] 预处理失败：{img_path} | 错误：{e}")
        traceback.print_exc()
        return None, None, None

def get_rpn_proposals(model, img_tensor, data_sample):
    try:
        model.eval()
        with torch.no_grad():
            feats = model.extract_feat(img_tensor.unsqueeze(0))
            rpn_results_list = model.rpn_head.predict(feats, [data_sample], rescale=False)
            instances = InstanceData()
            instances.bboxes = rpn_results_list[0].bboxes
            instances.scores = rpn_results_list[0].scores
            instances.labels = torch.zeros_like(instances.scores, dtype=torch.long)
            return instances
    except Exception as e:
        print(f"[ERROR] 获取 RPN 提案失败：{e}")
        traceback.print_exc()
        return None

def visualize_result(img, instances_or_result, out_file, name='proposals', score_thr=0.3):
    if instances_or_result is None:
        print(f"[WARNING] 跳过无效结果：{out_file}")
        return
    try:
        if isinstance(instances_or_result, DetDataSample):
            data_sample = instances_or_result
        else:
            data_sample = DetDataSample()
            data_sample.pred_instances = instances_or_result

        visualizer.add_datasample(
            name=name,
            image=img,
            data_sample=data_sample,
            draw_gt=False,
            show=False,
            out_file=out_file,
            pred_score_thr=score_thr
        )
        print(f"[INFO] 可视化已保存：{out_file}")
    except Exception as e:
        print(f"[ERROR] 可视化失败：{out_file} | 错误：{e}")
        traceback.print_exc()

def main():
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    if not img_files:
        print(f"[ERROR] 未找到输入图片，请放入 JPG 图像至 {input_dir}")
        return

    selected_imgs = random.sample(img_files, min(4, len(img_files)))

    for img_file in selected_imgs:
        img_path = os.path.join(input_dir, img_file)
        img_id = os.path.splitext(img_file)[0]

        img_tensor, data_sample, img = preprocess_image(img_path)
        if img_tensor is None:
            continue

        # RPN 提案
        rpn_instances = get_rpn_proposals(model, img_tensor, data_sample)
        visualize_result(img, rpn_instances, os.path.join(proposals_dir, f'{img_id}_proposals.jpg'), name='proposals')

        # 最终预测
        try:
            result = inference_detector(model, img_path)
            visualize_result(img, result, os.path.join(predictions_dir, f'{img_id}_predictions.jpg'), name='predictions')
        except Exception as e:
            print(f"[ERROR] 推理失败：{img_path} | 错误：{e}")
            traceback.print_exc()

if __name__ == '__main__':
    main()