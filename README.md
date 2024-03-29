OoDAnalyzer
==================================================================

Codes for the interactive analysis system, OoDAnalyzer, described in our paper ["OoDAnalyzer: Interactive Analysis of Out-of-Distribution Samples"](https://ieeexplore.ieee.org/document/8994105) (TVCG 2021).

~~Online demo: http://visgroup.thss.tsinghua.edu.cn:8183/~~

Requirements
----------
```
anytree==2.8.0
cffi==1.14.0
fastlapjv==1.0.0
Flask==1.1.2
matplotlib==3.1.3
numpy==1.18.4
Pillow==7.1.2
scikit-learn==0.22.1
scipy==1.4.1
```
Tested on Windows.

Usage Example
-----
Step 1: create a folder `data/` in the root folder.

Step 2: download demo data from Baiduyun(Link: [here](https://pan.baidu.com/s/1kFXlgW3pogn2NfSkw2Vyrw), password: 7nen) or Google Drive (Link: [here](https://drive.google.com/file/d/1-QP8DVa5dwOXgFVejassJUlRzYLK5v7o/view?usp=sharing), no password), and unpack it in the folder `data/`.

Step 3: setup the system:
```
python server.py
```

Step 4: visit http://localhost:8183/ with a browser.


## Citation
If you use this code for your research, please consider citing:
```
@article{chen2021oodanalyzer,
  author={Chen, Changjian and Yuan, Jun and Lu, Yafeng and Liu, Yang and Su, Hang and Yuan, Songtao and Liu, Shixia},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={{OoDAnalyzer}: Interactive Analysis of Out-of-Distribution Samples}, 
  year={2021},
  volume={27},
  number={7},
  pages={3335-3349}}
```

## Contact
If you have any problem about our code, feel free to contact
- ccj17@mails.tsinghua.edu.cn

or describe your problem in Issues.
