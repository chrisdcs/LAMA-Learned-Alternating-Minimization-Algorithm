# LAMA: Learned Alternating Minimization Algorithm for Dual-domain Sparse-View CT Reconstruction

This is the official site for the implementation of the published paper in MICCAI 2023: "Learned Alternating Minimization Algorithm for Dual-domain Sparse-View CT Reconstruction."

Here is the link for the [paper](https://arxiv.org/pdf/2306.02644.pdf)

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
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/iteration.jpg" width="960" />
</p>
Network unrolling:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/network.jpg" width="800" />
</p>

## Results
Mayo clinic LDCT grand challenge:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/mayoResult.png" width="720" />
</p>

|Metrics|Views|FBP|DDNet|LDA|DuDoTrans|Learn++|LAMA (ours)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|PSNR|64<br>128|27.17 ± 1.11<br>33.28 ± 0.85|35.70 ± 1.50<br>42.73 ± 1.08|37.16 ± 1.33<br>43.00 ± 0.91|37.90 ± 1.44<br>43.48 ± 1.04|43.02 ± 2.08<br>49.77 ± 0.96|44.58 ± 1.15<br>50.01 ± 0.69|
|SSIM|64<br>128|0.596 ± 9e−4<br>0.759 ± 1e−3|0.923 ± 4e−5<br>0.974 ± 4e−5|0.932 ± 1e−4<br>0.976 ± 2e−5|0.952 ± 1.0e−4<br>0.985 ± 1e−5|0.980 ± 3e−5<br>0.995 ± 1e−6|0.986 ± 7e−6<br>0.995 ± 6e−7|

NBIA dataset:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/NBIAResult.png" width="720" />
</p>

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
