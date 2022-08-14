OoDAnalyzer
==================================================================

Codes for the interactive analysis system, OoDAnalyzer, described in our paper ["A Unified Understanding of Deep NLP Models for Text Classification
"](https://ieeexplore.ieee.org/document/9801603) (TVCG 2022).

Online demo: http://nlpvis-demo.thuvis.org
Online video: http://nlpvis.thuvis.org/Video/nlpvis.mp4

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
Step 1: Install all the requirements.

Step 2: download text data (Link: [here](https://drive.google.com/file/d/1mwnl-Lr8xNsLWmBPmCOxHPKJtcCTdjwL/view?usp=sharing)), measure data (Link: [here](https://drive.google.com/file/d/18m4RxJnK0sBKunCt21TxsrrkXvLFz7ui/view?usp=sharing)), and polarity data (Link: [here](https://drive.google.com/file/d/1Ier_nqwKD3e8nD-4qSLjAXT5U-CjqxcI/view?usp=sharing)) from Google Drive , and unpack it in the root folder.
```
unzip dataset.zip
unzip cache.zip
unzip data.zip
```

Step 3: setup the system:
```
export FLASK_APP=app.py
flask run --port 5004
```

Step 4: visit http://localhost:5004/ with a browser.


## Citation
If you use this code for your research, please consider citing:
```
@ARTICLE{li2022nlpvis,
  author={Li, Zhen and Wang, Xiting and Yang, Weikai and Wu, Jing and Zhang, Zhengyan and Liu, Zhiyuan and Sun, Maosong and Zhang, Hui and Liu, Shixia},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={A Unified Understanding of Deep NLP Models for Text Classification}, 
  year={2022},
  pages={1-14},
  doi={10.1109/TVCG.2022.3184186}
}
```

## Contact
If you have any problem about our code, feel free to contact
- thu.lz@outlook.com

or describe your problem in Issues.
