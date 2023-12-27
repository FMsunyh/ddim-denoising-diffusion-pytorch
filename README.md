# ddim-denoising-diffusion-pytorch

复现论文 Denoising Diffusion Implicit Models, in Pytorch。

- 论文 | <a href="https://arxiv.org/abs/2010.02502">Denoising Diffusion Implicit Models</a> 

- 论文中的【Algorithm 1 Training】和【Algorithm 2 Sampling】公式推导 | https://blog.csdn.net/u010006102/article/details/134648877
- 复现代码 | https://github.com/FMsunyh/denoising-diffusion-pytorch


## python环境
- torch 1.13.0
- python 3.10
 
## 训练数据
- celeba数据集 | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
 
 百度云盘下载: CelebA/Img/img_align_celeba.zip

```
cp img_align_celeba.zip ./data/celebA/
cd ./data/celebA/
unzip img_align_celeba.zip
```

## 开启训练
- MNIST
```
python train.py --dataset mnist --epochs 6 --channels 1
```

- celebA
```
python train.py --dataset CelebA --epochs 100 --channels 3
```

## 输出路径
可以查看每一轮的预测结果
```
./outputs
```

## 测试效果


- MNIST数据集，训练6轮后的测试效果
 
| ![Alt text](demo/MNIST/sample_0.png) | ![Alt text](demo/MNIST/sample_1.png) | ![Alt text](demo/MNIST/sample_2.png)  | ![Alt text](demo/MNIST/sample_3.png)  | ![Alt text](demo/MNIST/sample_4.png)  | ![Alt text](demo/MNIST/sample_5.png)  | ![Alt text](demo/MNIST/sample_6.png)  | ![Alt text](demo/MNIST/sample_7.png)  |
| ------------------------------------ | ------------------------------------ | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- |
| ![Alt text](demo/MNIST/sample_8.png) | ![Alt text](demo/MNIST/sample_9.png) | ![Alt text](demo/MNIST/sample_10.png) | ![Alt text](demo/MNIST/sample_11.png) | ![Alt text](demo/MNIST/sample_12.png) | ![Alt text](demo/MNIST/sample_13.png) | ![Alt text](demo/MNIST/sample_14.png) | ![Alt text](demo/MNIST/sample_15.png) |

- celebA数据集，训练50轮后的测试效果

| ![Alt text](demo/CelebA/sample_0.png) | ![Alt text](demo/CelebA/sample_1.png) | ![Alt text](demo/CelebA/sample_2.png)  | ![Alt text](demo/CelebA/sample_3.png)  | ![Alt text](demo/CelebA/sample_4.png)  | ![Alt text](demo/CelebA/sample_5.png)  | ![Alt text](demo/CelebA/sample_6.png)  | ![Alt text](demo/CelebA/sample_7.png)  |
| ------------------------------------- | ------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| ![Alt text](demo/CelebA/sample_8.png) | ![Alt text](demo/CelebA/sample_9.png) | ![Alt text](demo/CelebA/sample_10.png) | ![Alt text](demo/CelebA/sample_11.png) | ![Alt text](demo/CelebA/sample_12.png) | ![Alt text](demo/CelebA/sample_13.png) | ![Alt text](demo/CelebA/sample_14.png) | ![Alt text](demo/CelebA/sample_15.png) |



## 参考
- 官方代码 | https://github.com/ermongroup/ddim/tree/main
- Denoising Diffusion Implicit Models (DDIM) Sampling | https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html
- labmlai | https://github.com/labmlai/annotated_deep_learning_paper_implementations
- CelebA Dataset | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- U-Net model for Denoising Diffusion Probabilistic Models (DDPM) | https://nn.labml.ai/diffusion/ddpm/unet.html

## AIGC学习交流
![Alt text](images/vx.png)