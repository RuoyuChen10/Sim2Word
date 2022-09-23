# Framework of Explaination in Face Verification

We compare with two framework: `Explainable Face Recognition (XFR)` and `xCos`

## XFR

xfr: [https://github.com/stresearch/xfr/](https://github.com/stresearch/xfr/)

please refer [test.py](xfr/demo/test.py) and [test_whitebox.py](xfr/demo/test_whitebox.py), which is ours replementation.

| Probe | Mate | Non-mate | CAM |
| -----|-|-|-|
| <img src="images/1/probe.jpg" width="112px"> | <img src="images/1/mate.jpg" width="112px"> | <img src="images/1/non-mate.jpg" width="112px"> | <img src="images/1/cam.jpg" width="112px"> | 
| <img src="images/2/probe.jpg" width="112px"> | <img src="images/2/mate.jpg" width="112px"> | <img src="images/2/non-mate.jpg" width="112px"> | <img src="images/2/cam.jpg" width="112px"> | 


## xCos

xCos: [https://github.com/ntubiolin/xcos](https://github.com/ntubiolin/xcos)

please refer `AUC.py`, `Visualization.py`, etc., for ours implementation.

Results:

| | |
|-|-|
| ![](./images/6.jpg) | ![](./images/7.jpg) |
| ![](./images/8.jpg) | ![](./images/9.jpg) |

cite:

```
@inproceedings{williford2020explainable,
  title={Explainable Face Recognition},
  author={Williford, Jonathan R and May, Brandon B and Byrne, Jeffrey},
  booktitle={European Conference on Computer Vision},
  pages={248--263},
  year={2020},
  organization={Springer}
}
```

```
@article{Lin2020xCosAE,
  title={xCos: An Explainable Cosine Metric for Face Verification Task},
  author={Yu-sheng Lin and Zheyu Liu and Yuan Chen and Yu-Siang Wang and Hsin-Ying Lee and Yirong Chen and Ya-Liang Chang and Winston H. Hsu},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.05383}
}
```