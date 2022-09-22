# Sim2Word

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7.1](https://img.shields.io/badge/pytorch-1.7.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-Apache_2.0-green.svg?style=plastic)

The official implementation of our work: Sim2Word: Explaining Similarity with Representative Attribute Words via Counterfactual Explanations

Code will be released soon.

## Pre-train Model

In our work we use 3 types of model for evaluation, `VGGFace Net`, `ArcFace Net`, and `CosFace Net`.

### VGGFace Net

First download the pre-trained model, the model is convert from coffee version from official file, don't worry it doesn't work.

| Arch type `VGGFace Net`         |                        download link                         |
| :----------------- | :----------------------------------------------------------: |
| `resnet50_ft`      | [link](https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU) |
| `senet50_ft`       | [link](https://drive.google.com/open?id=1YtAtL7Amsm-fZoPQGF4hJBC9ijjjwiMk) |
| `resnet50_scratch` | [link](https://drive.google.com/open?id=1gy9OJlVfBulWkIEnZhGpOLu084RgHw39) |
| `senet50_scratch`  | [link](https://drive.google.com/open?id=11Xo4tKir1KF8GdaTCMSbEQ9N4LhshJNP) |

Download model is `pkl` files.

### ArcFace and CosFace checkpoints

The pre-trained model of ArcFace and CosFace is from a project on github: [insightface](https://github.com/deepinsight/insightface)

- The models are available for non-commercial research purposes only.  
- All models can be found in here.  
- [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw  
- [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d)

If the above link is down, you can visit project insightface to find their checkpoints.

### Uncertainty model

The uncertainty model mentioned in our paper is also released, the implementation is from [https://github.com/RuoyuChen10/FaceTechnologyTool/FaceRecognition](https://github.com/RuoyuChen10/FaceTechnologyTool/tree/master/FaceRecognition). You can train your own model using this code.

![](https://github.com/RuoyuChen10/FaceTechnologyTool/blob/master/FaceRecognition/images/EDL.jpg)

### Self-training model

If you want to train your own face recognition model, you can refer our work from [https://github.com/RuoyuChen10/FaceTechnologyTool/FaceRecognition](https://github.com/RuoyuChen10/FaceTechnologyTool/tree/master/FaceRecognition), you can try:

- ArcFace Loss
- CosFace Loss
- Softmax Loss

## Acknowledgement

If you find this work is helpful for your research, please consider cite our work:

```bibtex
@article{chen2022sim2word,
  title   = {Sim2Word: Explaining Similarity with Representative Attribute Words via Counterfactual Explanations},
  author  = {Chen, Ruoyu and Li, Jingzhi and Zhang, Hua and Sheng, Changchong and Liu, Li and Cao, Xiaochun},
  journal = {ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  year    = 2022,
  publisher={ACM New York, NY}
}
```
