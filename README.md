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
1. For anaconda:
We provide a conda `environment.yml` file,
By this line code,
```
conda env create -f environment.yml
```
you can create a new conda environment including all dependence for 2-map.

then,
```
source activate 2-map
```
to enter this environment.

2. For other:
Package we need: 
  - matplotlib
  - numba
  - numpy
  - python=3.7.3
  - scikit-learn
  - scipy
  - umap-learn
  - jupyterlab(For experiment)

# Code Structure
* data folder: saving data which be used in our experiment
* exp folder: have some jupyter shell which is our experiment, and can be runned. The result will be store in exp_result folder in ./exp folder.
* utils: our 2-map package and some drawing function.

# Usage

For re-run our experiment, run .ipynb scipt on jupyter notebook(or jupyter lab) in exp folder.

- MNIST: It is umap experiment on MNIST dataset, for figure 1 & 2 in paper.
- fake_data: It is umap experiment on a fake dataset, a straight line in 100-dimension with four Gaussian disturbution
- FC_GAP: It is umap experiment on CAR dataset(trained on Res-50 with NPair loss), FC layer output vector with global pooling layer vector.
- ViCo_Glove: It is umap experiment in word embedding task, GloVe word vector with ViCo word vector.
- Training_process: It is umap experiment for time sequence vector, on CAR dataset(trained on Res-50 with NPair loss), for vectors after each epoch. 

The result will be shown in exp/exp_result/
