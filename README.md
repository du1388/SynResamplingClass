# Synthetic Resampling Techniques for Classification Modelling

### Introduction
This is a python script to conduct analysis on the impact of synthetic resampling techniques on the performance of binary classification models. For more information please check out our publication.

### Installation
We strongly recommend using Anaconda 3 enviroments which can be downloaded in from their [website](https://www.anaconda.com/distribution/#download-section). All requirements are given in ```requirements.txt```. 
For example from the anaconda3 prompt:
```
(base) C:\> cd path-to-this-directory
(base) C:\path-to-this-directory> conda create -n resample python=3.6
(base) C:\path-to-this-directory> conda activate resample
(resample) C:\path-to-this-directory> pip install -r requirements.txt
```
### Usage
For usage of the script please use the help argument:
```
python SynResampleClass.py --help
```
We provided two example datasets in the ```examples\``` directory. It is important to set the ```--target``` and ```--index``` headers as the same as the header in the data csv. For example:
```
python SynResampleClass.py examples\TextureSession_DFS_v2.csv result_output --target DFS --index ID
```

If there are any problems or questions, please email du94@hku.hk
