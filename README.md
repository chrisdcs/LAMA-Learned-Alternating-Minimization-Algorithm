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

## Dataset
The two dataset used for experiments in the paper are provided in this [OneDrive](https://uflorida-my.sharepoint.com/:u:/g/personal/ding_chi_ufl_edu/ETaqu5-bebZOjr0Bws3VxU4BkOVqJcwK_M0AvvcCzsqfgQ?e=umy9G0). To download this for Linux, you can use the following commands:
```
wget -O dataset.zip "https://uflorida-my.sharepoint.com/:u:/g/personal/ding_chi_ufl_edu/ETaqu5-bebZOjr0Bws3VxU4BkOVqJcwK_M0AvvcCzsqfgQ?e=umy9G0&download=1"
```
To unzip and process the data, you can use the following commands:
```
unzip dataset.zip
python process-data.py
```

(This link only has raw data, they need to be properly processed for training and testing in LAMA. More details coming soon.)

## Training (Cleaning up Code & Coming soon)

## Demo (Cleaning up Code & Coming soon)
CTLIB-demo.ipynb is ready.

## Model
LAMA in one iteration:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/iteration.jpg" width="800" />
</p>
Network unrolling:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/network.jpg" width="600" />
</p>

## Sparse-View CT Reconstruction
Mayo clinic LDCT grand challenge:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/mayoResult.png" width="720" />
</p>

NBIA dataset:
<p align="center">
  <img src="https://github.com/chrisdcs/LAMA-Learned-Alternating-Minimization-Algorithm/blob/master/figures/NBIAResult.png" width="720" />
</p>

## Experiments
Mayo:
| Metrics |  Views  |   FBP   |  DDNet  |   LDA   |DuDoTrans| Learn++ |LAMA(ours)|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|PSNR|64<br>128|27.17 ± 1.11<br>33.28 ± 0.85|35.70 ± 1.50<br>42.73 ± 1.08|37.16 ± 1.33<br>43.00 ± 0.91|37.90 ± 1.44<br>43.48 ± 1.04|43.02 ± 2.08<br>49.77 ± 0.96|44.58 ± 1.15<br>50.01 ± 0.69|
|SSIM|64<br>128|0.596 ± 9e−4<br>0.759 ± 1e−3|0.923 ± 4e−5<br>0.974 ± 4e−5|0.932 ± 1e−4<br>0.976 ± 2e−5|0.952 ± 1.0e−4<br>0.985 ± 1e−5|0.980 ± 3e−5<br>0.995 ± 1e−6|0.986 ± 7e−6<br>0.995 ± 6e−7|

NBIA:
| Metrics |  Views  |   FBP   |  DDNet  |   LDA   |DuDoTrans| Learn++ |LAMA(ours)|
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|PSNR|64<br>128|25.72 ± 1.93<br>31.86 ± 1.27|35.59 ± 2.76<br>40.23 ± 1.98|34.31 ± 2.20<br>40.26 ± 2.57|35.53 ± 2.63<br>40.67 ± 2.84|38.53 ± 3.41<br>43.35 ± 4.02|41.40 ± 3.54<br>45.20 ± 4.23|
|SSIM|64<br>128|0.592 ± 2e−3<br>0.743 ± 2e−3|0.920 ± 3e−4<br>0.961 ± 1e−4|0.896 ± 4e−4<br>0.963 ± 1e−4|0.938 ± 2e−4<br>0.976 ± 6e−5 |0.956 ± 2e−4<br>0.983 ± 5e−5|0.976 ± 8e−5<br>0.988 ± 3e−5|

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
