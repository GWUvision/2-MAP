import os
import numpy as np
import random
import matplotlib.pyplot as plt
import importlib
import datetime

def draw(emb1,emb2,label,name='',title1='',title2='',path='./',savename='comparison'):
    """ Draw a comparison image for two 2-d data

    Parameters
    ----------
    emb1:array of shape (n_samples,2)
        2-d data to be drawed.
    emb2:array of shape (n_samples,2)
        2-d data to be drawed.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    name: string(optional, default ''):
        The title of this comparison.
    title1: string(optional, default ''):
        title for first figure.
    title2: string(optional, default ''):
        title for second figure.
    path: string(optional, default './'):
        savepath.
    savename: string(optional, default 'comparison'):
        Savename for this comparison.

    Returns
    -------
    """

    size = len(label)
    a=label%7
    b =label%12
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","x","1","2","3","4"]
    colors = ['r','g','b','c','m', 'y', 'k']
    
    colorslist3=[]
    markerslist3 = []
    for item in a:
        colorslist3.append(colors[item])
    for item in b:
        markerslist3.append(markers[item])
        
    plt.figure(figsize=(20,10))
    plt.axis((-250,250,-250,250))
    plt.title(name)
    
    axe1=plt.subplot(1,2,1)
    axe1.set_title(title1)
    
    axe2=plt.subplot(1,2,2)
    axe2.set_title(title2)
    print(emb1.shape)
    print(size)
    x_max = np.max([np.max(emb1[:,0]), np.max(emb2[:, 0])])+2
    y_max = np.max([np.max(emb1[:,1]), np.max(emb2[:, 1])])+2
    x_min = np.min([np.min(emb1[:,0]), np.min(emb2[:, 0])])-2
    y_min = np.min([np.min(emb1[:,1]), np.min(emb2[:, 1])])-2
    axis_max = int(max(x_max, y_max))
    axis_min = int(min(x_min, y_min))
    for k in range(0,size-1):
        axe1.scatter(emb1[k, 0], emb1[k, 1],c=colorslist3[k],marker=markerslist3[k],alpha=0.5)
        axe1.set_xlim(axis_min, axis_max)
        axe1.set_ylim(axis_min, axis_max)
        axe2.scatter(emb2[k, 0], emb2[k, 1],c=colorslist3[k],marker=markerslist3[k],alpha=0.5)
        axe2.set_xlim(axis_min, axis_max)
        axe2.set_ylim(axis_min, axis_max)
        
    plt.savefig(path+'/'+savename+'.jpg',dpi=300)
    #plt.show()

def draw_single(emb,label,name,path='./',axis_size=35,savename=''):
    """ Draw a image for a 2-d data

    Parameters
    ----------
    emb:array of shape (n_samples,2)
        2-d data to be drawed.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    name: string(optional, default ''):
        The title of this figure.
    axis_size: int(optional, default ''):
        The size of axis.
    savename: string(optional, default ''):
        Savename for this comparison.

    Returns
    -------
    """
    size = len(label)
    a=label%7
    b =label%12
    markers = ["." , "," , "o" , "v" , "^" , "<", ">","x","1","2","3","4"]
    colors = ['r','g','b','c','m', 'y', 'k']
    
    colorslist3=[]
    markerslist3 = []
    for item in a:
        colorslist3.append(colors[item])
    for item in b:
        markerslist3.append(markers[item])
        
    plt.figure(figsize=(10,10))
    plt.axis((-250,250,-250,250))
    plt.title(name)
    
    axe1=plt.subplot(1,1,1)
    
    print(emb.shape)
    print(size)
    
    for k in range(0,size-1):
        axe1.scatter(emb[k, 0], emb[k, 1],c=colorslist3[k],marker=markerslist3[k],alpha=0.5)
        axe1.set_xlim(-axis_size, axis_size)
        axe1.set_ylim(-axis_size, axis_size)
        
    plt.savefig(path+'/'+savename+'.jpg',dpi=300)

def draw_curve(oriloss,Uloss,name='curve',savepath='./'):
    """ Draw accepted range and loss curve with different penalty scale.

    Parameters
    ----------
    oriloss:list of shape (times,)
        Loss for several times UMAP.
    Uloss: list
        loss for 2-MAP in different penalty scale.
    name: string(optional, default 'curve'):
        The title and savename of this figure.
    savepath: string(optional, default './'):
        Savepath for this curve figure.

    Returns
    -------
    """
    length = len(Uloss)
    x = range(-len(Uloss),0)
    a = [-len(Uloss),-1]
    mean = np.asarray(oriloss).mean()
    std = np.asarray(oriloss).std()
    Uloss.reverse()
    mean_line = [mean,mean]
    std_1_line = [mean+std,mean+std]
    std_2_line = [mean-std,mean-std]
    plt.figure(figsize=(8,4))
    plt.title(name)
    l1,=plt.plot(x,Uloss,linewidth=1)
    l2,=plt.plot(a,mean_line,c='r',linewidth=1)
    l3,=plt.plot(a,std_1_line,c='g',linewidth=1)
    l4,=plt.plot(a,std_2_line,c='g',linewidth=1)
    #c1,=plt.plot(x,yoked_KL_comp,c='b',linewidth=1)
    plt.legend(handles=[l1,l2,l3,l4], labels=['TUMAP cost','mean','mean+std','mean-std'],  loc='best')
    plt.xlabel("log(alpha)")
    plt.ylabel("Cost") 
    plt.savefig(savepath+name+'.png')
    plt.show()