import os
import numpy as np
import random
from . import tumap,Drawer
import pickle

def yoke_TUMAP(data1,data2,label,metric='euclidean',init_1="spectral",init_2="spectral",fixed=False,n_epoches=500,times=10,name1='embed1',name2='embed2',savepath='./',all_process=False,if_draw=True):
    """2-UMAP processes including parameter selection. Running several times UMAP to get
    accepted range and search for the accepted penalty scale.
    Parameters
    ----------
    data1:array of shape (n_samples,n_vector)
        high-dimensional data which will be visualized.
    data2:array of shape (n_samples,n_vector)
        high-dimensional data which will be visualized or low-dimensional fix map if fixed is setted as True.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    init_1: string (optional, default 'spectral')
        How to initialize the low dimensional embedding of data1. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    init_2: string (optional, default 'spectral')
        How to initialize the low dimensional embedding of data2. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    fixed: Boolean (optional, default False):
        A parameter to decide whether fixed second map. If True, 
        data2 should be set as a low-dimensional fix map.
        Making data1 yoked to fix map.
    n_epochs: int (optional, default 500)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    times: int (optional, default 10):
        The number of times to running original UMAP.
    name1: string(optional, default 'emb1'):
        Saving name and showing name in figure for data 1.
    name2: string(optional, default 'emb2'):
        Saving name and showing name in figure for data 2.
    savepath: string(optional, default './'):
        Saveing path to save result
    all_process: Boolean(optional, default 'False'):
        If find right parameter, whether continue generate for other small parameter.
    if_draw: Boolean(optional, default 'True'):
        Whether draw the 2-map comparison result.

    Returns
    -------
    result1: array of shape(n_samples,2)
        2-map result for data1.
    result2: array of shape(n_samples,2)
        2-map result for data2
    ori1: array of shape(n_samples,2)
        Umap result for data1
    ori2: array of shape(n_samples,2)
        Umap result for data2
    """

    oriloss1=[]
    oriloss2=[]
    
    ori1 = None
    ori2 = None
    for i in range(0,times):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=init_1,init_2=init_2,lam=0)
        embed1,embed2 = umaper.yoke_transform(data1,data2,fixed=fixed)
        if not os.path.isdir(savepath+'/ori/'):
            os.makedirs(savepath+'/ori/')
        pickle.dump( embed1, open( savepath+'/ori/'+name1+'_'+str(i)+'.pkl', "wb" ) )
        pickle.dump( embed2, open( savepath+'/ori/'+name2+'_'+str(i)+'.pkl', "wb" ) )
        loss1,loss2 = umaper.get_semi_loss()
        oriloss1.append(loss1)
        oriloss2.append(loss2)
        
    pickle.dump( oriloss1, open( savepath+'/ori/loss1.pkl', "wb" ) )
    pickle.dump( oriloss2, open( savepath+'/ori/loss2.pkl', "wb" ) )        
    ori1 = embed1
    ori2 = embed2

    mean1 = np.asarray(oriloss1).mean()
    std1 = np.asarray(oriloss1).std() 
    max1 = np.asarray(oriloss1).max() 
    mean2 = np.asarray(oriloss2).mean()
    std2 = np.asarray(oriloss2).std()
    max2 = np.asarray(oriloss2).max() 
    
    result1 = None
    result2 = None
    Uloss1=[]
    Uloss2=[]
    for i in range(1,10):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=init_1,init_2=init_2,lam=10**-i)
        embed1, embed2 = umaper.yoke_transform(data1,data2,fixed=fixed)
        if not os.path.isdir(savepath+'/yoked/'):
            os.makedirs(savepath+'/yoked/')
        pickle.dump( embed1, open( savepath+'/yoked/'+name1+'_'+str(-i)+'.pkl', "wb" ) )
        pickle.dump( embed2, open( savepath+'/yoked/'+name2+'_'+str(-i)+'.pkl', "wb" ) )
        loss1,loss2 = umaper.get_semi_loss()
        Uloss1.append(loss1)
        Uloss2.append(loss2)
        condition1 = (loss1+loss2<=mean1+std1+mean2+std2)
        if not all_process:
            if fixed is False:
                if condition1:
                    res1 = embed1
                    res2 = embed2
                    result1 = res1
                    result2 = res2
                    break
            else:
                if loss1<=mean1+std1:
                    res1 = embed1
                    res2 = embed2
                    result1 = res1
                    result2 = res2
                    break
        else:
            if fixed is False:
                if condition1:
                    res1 = embed1
                    res2 = embed2
                    if result1 is None:
                        result1 = res1
                        result2 = res2
            else:
                if loss1<=mean1+std1:
                    result1 = embed1
                    result2 = embed2
                    if result1 is None:
                        result1 = res1
                        result2 = res2
    pickle.dump(oriloss1, open( savepath+'/yoked/loss1.pkl', "wb" ) )
    pickle.dump(oriloss2, open( savepath+'/yoked/loss2.pkl', "wb" ) )
    oriloss=list(np.asarray(oriloss1)+np.asarray(oriloss2))
    Uloss=list(np.asarray(Uloss1)+np.asarray(Uloss2))
    Drawer.draw_curve(oriloss,Uloss,name='loss_curve',savepath=savepath)
    if if_draw:
        Drawer.draw(result1,result2,label,'',name1,name2,savepath=savepath,savename='comparison')
    return result1,result2,ori1,ori2

def ThruMap(datalist,label,metric='euclidean',n_epoches=500,times=5,savepath='./',if_draw=True):
    """ThruMAP processes including parameter selection. To visulize a series of datas. Running 2-map in order,
    fixing previous 2-map result, and align new one to previous one. Running several times UMAP to get 
    accepted range and search for the accepted penalty scale.
    Parameters
    ----------
    datalist:list of array, shape:(n_samples,n_vector,n)
        high-dimensional data list which will be visualized.
    label: list of shape (n_samples,)
        label for data, used to draw visualization.
    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:
            * euclidean
            * manhattan
            * chebyshev
            * minkowski
            * canberra
            * braycurtis
            * mahalanobis
            * wminkowski
            * seuclidean
            * cosine
            * correlation
            * haversine
            * hamming
            * jaccard
            * dice
            * russelrao
            * kulsinski
            * rogerstanimoto
            * sokalmichener
            * sokalsneath
            * yule
        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.
    n_epochs: int (optional, default 500)
        The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If None is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    times: int (optional, default 10):
        The number of times to running original UMAP.
    savepath: string(optional, default './'):
        Saveing path to save result
    if_draw: Boolean(optional, default 'True'):
        Whether draw the 2-map comparison result.

    Returns
    -------
    result1: array of shape(n_samples,2)
        2-map result for data1.
    result2: array of shape(n_samples,2)
        2-map result for data2
    ori1: array of shape(n_samples,2)
        Umap result for data1
    ori2: array of shape(n_samples,2)
        Umap result for data2
    """    
    data1 = datalist[0]
    data2 = datalist[1]
    
    oriloss1=[]
    oriloss2=[]
    
    ori1 = None
    ori2 = None
    
    for i in range(0,times):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,lam=0)
        embed1,embed2 = umaper.yoke_transform(data1,data2,fixed=False)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        loss1,loss2 = umaper.get_semi_loss()
        oriloss1.append(loss1)
        oriloss2.append(loss2)

    mean1 = np.asarray(oriloss1).mean()
    std1 = np.asarray(oriloss1).std() 
    max1 = np.asarray(oriloss1).max() 
    mean2 = np.asarray(oriloss2).mean()
    std2 = np.asarray(oriloss2).std()
    max2 = np.asarray(oriloss2).max() 
    
    result1 = None
    result2 = None
    for i in range(1,10):
        umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,lam=10**-i)
        embed1, embed2 = umaper.yoke_transform(data1,data2,fixed=False)
        loss1,loss2 = umaper.get_semi_loss()
        condition1 = (loss1+loss2<=mean1+std1+mean2+std2)
        if condition1:
            res1 = embed1
            res2 = embed2
            result1 = res1
            result2 = res2
            break
    pickle.dump(result1, open(savepath+'/0.pkl', "wb" ) )
    pickle.dump(result2, open(savepath+'/1.pkl', "wb" ) )
    if if_draw:
        Drawer.draw_single(result1,label,'',savepath=savepath,savename='0')
        Drawer.draw_single(result2,label,'',savepath=savepath,savename='1')
        
    fixmap=result2
    for index in range(2,len(datalist)):
        data = datalist[index]
        for i in range(0,times):
            umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=fixmap,init_2=fixmap,lam=0)
            embed1,embed2 = umaper.yoke_transform(data1,fixmap,fixed=True)
            loss1,loss2 = umaper.get_semi_loss()
            oriloss1.append(loss1)
            oriloss2.append(loss2)
            mean1 = np.asarray(oriloss1).mean()
            std1 = np.asarray(oriloss1).std() 
            max1 = np.asarray(oriloss1).max() 
            mean2 = np.asarray(oriloss2).mean()
            std2 = np.asarray(oriloss2).std()
            max2 = np.asarray(oriloss2).max() 

            result1 = None
            result2 = None

        for i in range(1,10):
            umaper = tumap.UMAP(metric=metric,n_epochs=n_epoches,init_1=fixmap,init_2=fixmap,lam=10**-i)
            embed1, embed2 = umaper.yoke_transform(data1,fixmap,fixed=True)

            loss1,loss2 = umaper.get_semi_loss()
            condition1 = (loss1<=mean1+std1)
            if condition1:
                res1 = embed1
                res2 = embed2
                result1 = res1
                result2 = res2
                break
        pickle.dump(result1, open(savepath+'/'+str(index)+'.pkl', "wb" ) )
        fixmap=result1    
        if if_draw:
            Drawer.draw_single(result1,label,'',savepath=savepath,savename=str(index))
        
    return result1,result2,ori1,ori2