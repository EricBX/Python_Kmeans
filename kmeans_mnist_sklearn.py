# K-Means for MNIST dataset based on sklearn
# Author: EricBX
# Date: 2020.4

import os
import time

import cv2
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

from decompress_mnist import decompress
from utils_cluster import findOptimalFeatureDimension, findOptimalK, \
    visualize_features, visualize_kmeans_result, saveResults, evaluateKMeans

# 预设参数
# MNIST文件目录
data_dir = "./mnist"
# 保存MNIST的文件名
result_file_name = 'decompress_mnist'
# 选择使用test或train做实验
data_phase = 'train'
# 输出文件夹
output_path = './result'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
# 选择预处理方式
preprocess = 'LDA'  # PCA 或 LDA
# 保留的特征维度数
feature_dimension = 30
# 采样的随机种子
random_seed = 6 # 实验报告基于 random_seed = 6
# 每一类使用的样本数
n_samples = 100 # 太多可视化效果不好
# 初始化方法
init_method = 'k-means++' # 'k-means++' 或者 'random'


if __name__ == '__main__':
    # 数据准备：解压mnist    # 只需运行一次
    decompress(data_dir, result_file_name)

    # 数据读取：从test图片中取1k张图片，每一类包括100张图片
    data = []  # 保存图片像素值及其标签
    for i in range(10):
        filepath = os.path.join(data_dir, result_file_name, data_phase, '%d' % i)
        img_total_num = len(os.listdir(filepath))
        for j in range(n_samples):
            img_num = (j * 10 + random_seed) % img_total_num  # 每隔十张取一张图片
            filename = os.path.join(filepath, '%d.png' % img_num)
            img_vec = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).flatten()
            data.append(np.hstack((img_vec, i)))
    print('data shape =', np.shape(data))
    data_raw = np.array(data)[:, :-1]  # 只保留图片像素值，便于接下来进行PCA降维处理
    labels_gt = np.array(data)[:, -1]  # 取标签真值，用作可视化，以及定量分析聚类效果

    # 数据预处理：归一化
    data_std = StandardScaler().fit_transform(data_raw)

    # 数据预处理：降维
    features = None
    if preprocess == 'PCA':
        # PCA降维    # 取前n个特征向量，即，每张图保留n个特征
        pca = PCA(feature_dimension)
        pca.fit(data_std)
        features = pca.transform(data_std)

        # 分析PCA降维后的特征维度的影响
        findOptimalFeatureDimension(data_std, labels_gt, 1, 30, 'PCA', output_path)

        # # 特征向量可视化（略）
        # eigen_values = pca.components_.reshape(feature_dimension, 28, 28)
        # visualize_features(feature_dimension, eigen_values)

    elif preprocess == 'LDA':
        # LDA降维 # 借助标签，选择分类性能最好的方向（临时替代特征提取）
        lda = LDA(n_components=feature_dimension)
        features = lda.fit_transform(data_std, labels_gt)

        # 分析LDA降维后的特征维度的影响
        findOptimalFeatureDimension(data_std, labels_gt, 1, 30, 'LDA', output_path)

    # 超参数选择：利用 elbow method 对 k 调参    # 尽管我们已经知道，k应该取10
    findOptimalK(features, 1, 15, output_path)

    # k-means 聚类
    k_clusters = 10
    t1 = time.time()
    kmeans = KMeans(n_clusters=k_clusters,  # fixed
                    init=init_method,  # to be tuned
                    # n_init=10, # default trick for precision
                    # max_iter=300, # default is enough
                    # tol=1e-4, # default is enough
                    precompute_distances=True,  # trick for speed
                    n_jobs=-1,  # trick for speed
                    algorithm='full'  # use classical EM-style algorithm (though not efficient)
                    )
    kmeans = kmeans.fit(features)
    t2 = time.time()
    labels_cluster = kmeans.labels_

    # 聚类结果可视化一：取前两维特征，作图
    visualize_kmeans_result(kmeans, features, labels_gt, output_path)

    # 聚类结果可视化二：每一簇包含的图片（最多）取100张，保存
    saveResults(labels_cluster, data_raw, output_path)

    # 定量评价：根据标签真值，计算精确率 precision
    evaluateKMeans(labels_cluster, labels_gt)

    # 定量评价：根据标签真值，计算精确率 precision
    Precision = evaluateKMeans(labels_cluster, labels_gt)


    # 结果输出
    print('time cost = %.4f' % (t2 - t1))
    print('number of iteration = %d' % kmeans.n_iter_)
    print('sum of squared distances = %d' % kmeans.inertia_)
    SSE = kmeans.inertia_

    # Evaluation: CH score
    CH = metrics.calinski_harabaz_score(labels_cluster[:, np.newaxis], labels_gt)
    print("KMeans clustering CH score = %.4f " % CH)

    # Evaluation: Silhouette score
    SS = metrics.silhouette_score(labels_cluster[:, np.newaxis], labels_gt, metric='euclidean')
    print("KMeans clustering Silhouette score = %.4f " % SS)

    f = open('./log.txt','a')
    f.write('\n')
    f.write(time.asctime( time.localtime(time.time()) ))
    f.write('\n SSE = %f \n Prec = %f \n CH = %f \n SS = %f'% (SSE, Precision, CH, SS))
    f.write('\n')
