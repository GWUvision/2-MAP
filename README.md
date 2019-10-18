# 2-MAP
It is an implementation for our WACV paper: 2-MAP: Aligned Visualizations for Comparison of High-Dimensional Point Sets. Link: TODO

# Contents
- [Overview](#overview)
- [Install Dependencies](#install-dependencies)
- [Code Structure](#code-structure)
    - [Directories](#directories)
- [Usage](#Usage)

# Overview
Welcome to our project  2-MAP: Aligned Visualizations for Comparison of High-Dimensional Point Sets.

**Available on Arxiv**: TODO

# Install Dependencies

We provide a conda `environment.yml` file,
By this line code,
```
conda env create -f environment.yml
```
you can create a new conda environment including all dependence for 2-map.

# Code Structure
* data folder: saving data which be used in our experiment
* exp folder: have some jupyter shell which is our experiment, and can be runned. The result will be store in exp_result folder in ./exp folder.
* utils: our 2-map package and some drawing function.

# Usage:
For our 2-map, you can use:
```

pip install 2-map
```
to install our package.

And then, we have two main functions, first one is yoke_2map:
```
autoUMap.yoke_TUMAP(data1,data2,label,metric='euclidean',init_1="spectral",init_2="spectral",fixed=False,n_epoches=500,times=10,name1='embed1',name2='embed2',savepath='./',all_process=False,if_draw=True)
```
yoke_2map is used for comparison two high dimensional data, the detail can be found in our paper.

* data1 is the first high dimensional data
* data2 is the second high dimensional data, is fixed = True, data2 is the fix low dimensional map.
* label is label for data which is used to visualize two comparison
* metric shows how measure distance in high dimension, default 'euclidean', for other options, see comment in code.
* init_1,init2_ is initialization for map1 and map2, default 'spectral', for other options, see comment in code.
* fixed, default False, when be setted True, data1 will be align similar to fix low dimensional map data2.
* n_epoches, training epoch, default 500.
* times, default 10, how much we run UMAP for finding accepted penlty scale range.
* name1,name2 is savename for two embedding result, and title showed in comparison figure.
* savepath is savepath for result.
* all_process, whether run 2-map for whole 10 penalty scale.
* if_draw, whether draw comparison figure.


Second one is ThruMap:
```
autoUMap.ThruMap(datalist,label,metric='euclidean',n_epoches=500,times=5,savepath='./',if_draw=True)
```
ThruMap is used for align a series of high dimensional data.

* datalist is data series.
* label is label for data which is used to visualize.
* metric shows how measure distance in high dimension, default 'euclidean', for other options, see comment in code.
* n_epoches, training epoch, default 500.
* times, default 10, how much we run UMAP for finding accepted penlty scale range.
* savepath is savepath for result.
* if_draw, whether draw comparison figure.
