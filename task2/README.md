
## 环境配置

1. Python $3.8$
2. Pytorch $2.0.0$
3. CUDA $12.1$
4. Tensorboard $2.19.0$
5. mmcv $2.1.0$
6. mmengine $0.10.7$
7. MMedtection $3.3.0$

## 说明

### **数据集下载**

在 Linux 系统使用 aria2 下载 VOC2012 数据集并解压缩：
```bash
aria2c -x 16 -s 16 http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.1.tar
```
数据集结构如下：
```
data/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── ImageSets/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── SegmentationObject/
```

### **tools**

- **voc2coco.py**

将 VOC 数据集转化成可以直接用于训练的 COCO 格式，将数据集下载到 MMDetection 框架下后运行该程序即可。


- **train.py**

MMDetection 提供的模型训练脚本，运行如下指令：
```bash
python tools/train.py $配置文件.py$
```
其中配置文件在`work_dirs`目录中。

- **test.py**

MMDetection 提供的模型测试脚本，运行如下指令：
```bash
python tools/test.py $配置文件.py$ $模型权重.pth$ visual_results
```

- **visualize.py**

用于可视化对比训练好的 Mask R-CNN 第一阶段产生的 proposal box 和最终的预测结果，需要将输入图像保存在`visualization/in/`目录下，运行`visualize.py`，会把 proposal box 和最终结果分别保存至 `visualization/out/`目录下的`proposal/`和`prediction/`.

### **work_dirs**：工作目录

Mask R-CNN 和 Sparse R-CNN 两种模型的配置文件分别位于 `mask-rcnn_r50_fpn_1x_coco/`和`sparse-rcnn_r50_fpn_1x_coco/`目录下。
运行如下指令：
```bash
tensorboard logdir=work_dirs
```
可以看到训练和测试的 Tensorboard 可视化信息。

### **模型权重下载**

`mask_model.pth`和`sparse_model.pth`：可从 [百度网盘](https://pan.baidu.com/s/1U5eOwVkrgcDK2tz5RVnfng)下载， 提取码: **n4cm**
