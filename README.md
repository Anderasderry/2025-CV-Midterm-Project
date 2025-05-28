# 2025-CV-Midterm-Project

该项目为复旦大学2025年春季课程计算机视觉（DATA $130051.01$ ）期中作业，共包含两个任务，任务代码及报告分别位于task $1$ 与task $2$ 目录下，任务具体要求如下。

**任务1：**  
微调在 ImageNet 上预训练的卷积神经网络实现 Caltech $\textendash 101$ 分类

**基本要求：**
(1) 训练集测试集按照 [Caltech-101]( [https://data.caltech.edu/records/mzrjq-6wc02Links to an external site.](https://data.caltech.edu/records/mzrjq-6wc02)) 标准；  
(2) 修改现有的 CNN 架构（如 AlexNet, ResNet-18）用于 Caltech $\textendash 101$ 识别，通过将其输出层大小设置为 101 以适应数据集中的类别数量，其余层使用在ImageNet上预训练得到的网络参数进行初始化；  
(3) 在  Caltech $\textendash 101$ 数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；  
(4) 观察不同的超参数，如训练步数、学习率，及其不同组合带来的影响，并尽可能提升模型性能；  
(5) 与仅使用 Caltech $\textendash 101$ 数据集从随机初始化的网络参数开始训练得到的结果 **进行对比**，观察预训练带来的提升。
  
**任务2：**  
在VOC数据集上训练并测试模型 Mask R-CNN 和 Sparse R-CNN 

基本要求：  
（1） 学习使用现成的目标检测框架——如[mmdetection]([https://github.com/open-mmlab/mmdetectionLinks to an external site.](https://github.com/open-mmlab/mmdetection))——在VOC数据集上训练并测试目标检测模型Mask R-CNN 和Sparse R-CNN；  
（2） 挑选4张测试集中的图像，通过可视化**对比**训练好的Mask R-CNN第一阶段产生的proposal box和最终的预测结果，以及Mask R-CNN 和Sparse R-CNN的**实例分割**与**目标检测**可视化结果；  
（3） 搜集三张不在VOC数据集内包含有VOC中类别物体的图像，分别可视化并比较两个在VOC数据集上训练好的模型在这三张图片上的目标检测/实例分割结果（展示bounding box、instance mask、类别标签和得分）。
