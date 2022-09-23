# Sim2Word

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7.1](https://img.shields.io/badge/pytorch-1.7.1-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-Apache_2.0-green.svg?style=plastic)

The official implementation of our work: Sim2Word: Explaining Similarity with Representative Attribute Words via Counterfactual Explanations

![](images/Fig1.png)

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

## The frame work of interpreting similarity

![](images/Fig2.png)

It's paper's Section 3 implementation, please refer to `Scores-sameid.py`, or refer to `Multi_Identity_topk.py` (set `topk = 1`).

```shell
python Multi_Identity_topk.py
```

After that, you can get the results from fold [results](./results), you can get some visualization results from a fold called `cam` such as:

||||||
|-|-|-|-|-|
|![](results/example/attribute-o0.jpg)|![](results/example/attribute-o1.jpg)| ![](results/example/attribute-o2.jpg)|![](results/example/attribute-o3.jpg)|![](results/example/attribute-o4.jpg)|
|![](results/example/attribute0.jpg)|![](results/example/attribute1.jpg)|![](results/example/attribute2.jpg)|![](results/example/attribute3.jpg)|![](results/example/attribute4.jpg)|

another fold called `json` stores the experiment values, than you can use this to get quantitive results:

```
cd results
python AUC_most_attributes.py
```
Pay attention to modify the placeholders in the python file, such as `args.Json_fold`.

You can also use this to get the top-5 most representative attribute:

```shell
python Top_5_attributes.py
```

|||
| ------ | -|
|![](results/example/3.jpg)| ![](results/example/9.jpg) |
|![](results/example/4.jpg)| ![](results/example/5.jpg) |

## Interpreting Face Identification

![](images/Fig3.png)

This section refer to the Section 4 of the paper. And also the results of Fig. 9.

First, to get the inter results:

```shell
python quantity_single_person_uncertain.py
```

than:

```shell
cd results
python uncertainty_visualization.py
```

you can get some visualization like:

| | |
| - | - |
|![](results/example/n000002.jpg)| ![](results/example/n000003.jpg) |
|![](results/example/n000004.jpg)| ![](results/example/n000005.jpg) |

## Different Strategy

There also some strategies in our paper, which mentioned in Section 5.5. For the topk strategy please refer to [Multi_Identity_topk.py](Multi_Identity_topk.py), and threshold strategy refers to [Multi_identity_thresh.py](Multi_identity_thresh.py).

## Method Comparision

![](images/Fig4.png)

Please refer to fold [Explainable-Framework-Comparison](./Explainable-Framework-Comparison), and follow their command to implement the results.

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
