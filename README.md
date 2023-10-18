# LAMA: Learned Alternating Minimization Algorithm for Dual-domain Sparse-View CT Reconstruction

This is the official site for the implementation of the published paper in MICCAI 2023: "Learned Alternating Minimization Algorithm for Dual-domain Sparse-View CT Reconstruction."


## Dependencies
```
pytorch==1.10
scipy==1.11.1
numpy==1.25.2
scikit-image==0.19.3
opencv-python==4.8.0
```
Computational Tomography Library: [CTLIB](https://github.com/xwj01/CTLIB)

(More stuff like training, testing, demo, and dataset coming soon...)

## Model
LAMA in one iteration:
![](https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/iteration.jpg)
Network unrolling:
![](https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/network.jpg)

## Visual Results
![](https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/mayoResult.png)
![](https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/NBIAResult.png)

## Citation
If you find this implementation useful, please consider citing our work:
```bibtex
@InProceedings{ding2023LAMA,
  title={Learned Alternating Minimization Algorithm for Dual-Domain Sparse-View CT Reconstruction},
  author={Ding, Chi and Zhang, Qingchao and Wang, Ge and Ye, Xiaojing and Chen, Yunmei},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year={2023},
  publisher={Springer Nature Switzerland},
  pages={173--183},
  isbn={978-3-031-43999-5"}
}
```
