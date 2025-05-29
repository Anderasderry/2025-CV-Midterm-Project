## 任务一：微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类
### 环境配置

1. Python $3.11$
2. torch $2.0.0$
3. CUDA $11.8$
4. Tensorboard $2.19.0$


### 说明


- **数据集下载**
    
    从 [Caltech-101官网](https://data.caltech.edu/records/mzrjq-6wc02) 下载 `101_ObjectCategories.tar.gz`文件，解压后将文件夹放至项目目录，保持文件名为 `101_ObjectCategories/`。
    


- **main.py**
	
    训练主程序，对训练轮次，特征层学习率及输出层（全连接层）学习率三个超参数进行网格化搜索，利用 Tensorboard 可视化训练过程中在训练集和验证集上的loss曲线和验证集上的 accuracy 变化，并存储最优模型权重。
    


- **grid_search_results.csv**
	
	保存了`main.py`中每一组超参数训练的最优模型在验证集和测试机上的准确率。


- **train_from_scratch.py**
    
    不带预训练的对比实验，将超参数设置为预训练的模型在网格搜索中得到的最优组合，其余参数随机初始化，训练并存储最优模型权重。


- **TensorBoard可视化**
	
	训练日志已上传至 `runs/`目录下，在命令行中运行：
	```bash
	tensorboard --logdir=runs
	```
	默认打开 `http://localhost:6006`，可以看到所有记录的可视化信息。
	

- **模型权重下载**
    
	`best_model.pth`：经过预训练过的模型，在测试集上的准确率达到 96.33% . <br>
  `best_model_without_pretrained.pth`：未经预训练的模型，在测试机上的准确率达到 51.35% . <br>
    以上文件可从[百度网盘]( https://pan.baidu.com/s/1faKQQgeIJwx_gVba9orzkg)下载，提取码：**bj4k**.
    

- **model_eval.py**
	
	将训练好的模型权重保存至项目文件夹中，在命令行输入如下指令，可测试模型在测试集上的准确率。
	```bash
	\\测试预训练过的模型
	python model_eval.py --pth $文件名(如best_model.pth)$
	（或）
	\\测试随机初始化的模型
	python model_eval.py --pth $文件名(如best_model.pth)$ --no-pretrained
	```
