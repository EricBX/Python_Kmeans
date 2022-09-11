# K-Means based on numpy
# Author: EricBX
# Date: 2020.6

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

if __name__ == '__main__':
    # (0) 读取数据集A、B、C
    filepath = './samples'
    trainfile = os.path.join(filepath, 'clusteringA.train')
    with open(trainfile) as f:
        content = f.readlines()
    traindata = np.zeros((len(content),2))
    for i in range(len(content)):
        traindata[i,:] = content[i].split()

    # (1) 用K-Means算法聚类，K=4，前四个样本作为初始聚类中心
    MaxIter = 100
    iter = 0
    Z_trace = []
    labels = []
    t_begin = time.time()
    # step1 初始化
    K = 4
    n = np.shape(traindata)[0]
    dim = np.shape(traindata)[1]
    Z = np.zeros((K,dim))
    for i in range(K):
        Z[i] = traindata[i,:]
    Z_trace.append(Z)
    print('Clustering begin with Z = ')
    print(Z)
    for ii in range(MaxIter):
        iter += 1 # 认为每做一次样本划分和修正中心，迭代一次
        # step2 样本划分
        labels = np.zeros((n,1),dtype=int)
        for i in range(n):
            d = np.zeros((K,1))
            for j in range(K):
                d[j] = np.linalg.norm(traindata[i]-Z[j])
            labels[i] = np.argmin(d)
        # step3 修正中心
        Z_new = np.zeros((K,dim))
        Z_count = np.zeros((K,1))
        for i in range(n):
            Z_new[labels[i],:] += traindata[i]
            Z_count[labels[i]] += 1
        Z_new = Z_new / Z_count
        # step4 判断是否收敛
        if not (Z_new - Z).any():
            print('Iteration %d, Clustering Finished!'%iter)
            break
        else:
            Z = Z_new
            Z_trace.append(Z)
            print('Iteration %d, Z = '%(iter))
            print(Z)
    print('K-Means result:\nTotal iterations:%d\nTime:%.4fs'%(iter, (time.time()-t_begin)))
    print('Z = ')
    print(Z)

    # (2) 估计4类样本各自的数学期望，协方差矩阵
    means = Z
    for i in range(K):
        print('m%d = (%.4f, %.4f)'%(i+1,means[i,0], means[i,1]))
    vectors = {}
    for i in range(K):
        vectors[i] = traindata[np.where(labels==i)[0],:]
    Covs = np.zeros((K,dim,dim))
    for i in range(K):
        Covs[i] = np.cov(np.transpose(vectors[i])) # N-1
        # Covs[i] = np.dot(np.transpose(vectors[i]-means[i]),(vectors[i]-means[i]))/len(vectors[i]) # N
        print('C%d:'%(i+1))
        print(Covs[i])

    # (3) 可视化聚类结果，横纵坐标，数据范围，四种颜色，数学期望（点），协方差矩阵椭圆
    fig = plt.figure()
    ax = plt.subplot()
    ax.set_title('K-Means Clustering Result')
    colorlist = ['r', 'y', 'g', 'c']
    for i in range(K):
        # 画椭圆
        lambdas, alphas = np.linalg.eig(Covs[i,:,:])
        s = 4.605 # 根据置信区间，查卡方概率表： 90% 4.605 | 95% 5.991 | 99% 9.21
        height = 2 * np.sqrt(s * lambdas[0]) # 短轴
        width = 2 * np.sqrt(s * lambdas[1]) # 长轴
        angle = -np.arctan2(alphas[1,1], alphas[1,0]) # 最大特征值对应特征向量与x轴夹角（输入顺序：y,x）
        ellipse = Ellipse(means[i], width, height, angle*180/np.pi, edgecolor=colorlist[i], facecolor='none')
        ellipse.set_alpha(0.5)
        ax.add_patch(ellipse)
        # 画数据
        ax.scatter(vectors[i][:,0], vectors[i][:,1], s=5, c=colorlist[i], marker='o')
        ax.scatter(means[i][0], means[i][1], s=30, c='k', marker='*')
    # xlim = [np.min(traindata, 0)[0], np.max(traindata, 0)[0]]
    # ylim = [np.min(traindata, 0)[1], np.max(traindata, 0)[1]]
    # ax.set_xlim(xlim[0], xlim[1])
    ax.legend(['cluster1', 'cluster2', 'cluster3', 'cluster4'])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # plt.show()
    if not os.path.exists('./result'):
        os.mkdir('./result')
    plt.savefig('./result/visSamples.png')

