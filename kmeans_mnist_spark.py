# K-Means for MNIST dataset based on Spark architecture
# Author: EricBX
# Date: 2020.5

import os
import time

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# from decompress_mnist import decompress
from utils_cluster import findOptimalFeatureDimension, findOptimalK, \
    visualize_features, visualize_kmeans_result, saveResults, evaluateKMeans
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.types import Row
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import DenseVector
# from pyspark.ml.feature import StandardScaler
# from pyspark.ml.feature import PCA


# settings
# MNIST folder
data_dir = "./mnist"
# to save
result_file_name = 'decompress_mnist'
# use test or train
data_phase = 'train'
# output
output_path = './result'
if not os.path.isdir(output_path):
    os.mkdir(output_path)
# how many features to use
feature_dimension = 10
# for sampling
random_seed = 6 #  we choose 6
# sample number of each class
n_samples = 100 # we choose 100
# initial method
init_method = 'k-means||' # 'k-means||' or 'random'
# choose model
model = 'k-means'
# set distance function
distfunc = 'euclidean' # 'euclidean' or 'cosine'
# spark preparation
sc =SparkContext()
spark = SparkSession(sc)

if __name__ == '__main__':
    # data preparation: decompress MNIST
    # decompress(data_dir, result_file_name)

    # data reading: read 1k images from train set, each class 100
    data = []  # to save
    for i in range(10):
        filepath = os.path.join(data_dir, result_file_name, data_phase, '%d' % i)
        img_total_num = len(os.listdir(filepath))
        for j in range(n_samples):
            img_num = (j * 10 + random_seed) % img_total_num  # get 1 img for every 10 imgs
            filename = os.path.join(filepath, '%d.png' % img_num)
            img_vec = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).flatten()
            data.append(np.hstack((img_vec, i)))
    print('data shape =', np.shape(data))
    data_raw = np.array(data)[:, :-1]
    labels_gt = np.array(data)[:, -1]

    # data preprocessing: Standard Scaler
    data_std = StandardScaler().fit_transform(data_raw)

    # data prepocessing: LDA
    lda = LDA(n_components=feature_dimension)
    features = lda.fit_transform(data_std, labels_gt)
    features_with_labels = np.hstack((features, labels_gt[:,np.newaxis]))

    # transform to dataFrame
    rdd = sc.parallelize(features_with_labels, numSlices=n_samples * 10)
    dataFrame = rdd.map(lambda x: [float(i) for i in x]).map(
        lambda x: Row(label=x[-1], features=DenseVector(x[:-1]))).toDF()

    k_clusters = 10
    if model == 'k-means':
        # Trains a k-means model.
        print('training k-means ...')
        t_begin = time.time()
        kmeans = KMeans().setK(k_clusters).setSeed(1).setFeaturesCol('features').setPredictionCol('prediction')
        kmeans.setInitMode(init_method)
    else:
        # Trains a bisecting k-means model.
        print('training bisecting k-means ...')
        t_begin = time.time()
        kmeans = BisectingKMeans(k=k_clusters, minDivisibleClusterSize=1.0)
    kmeans.setDistanceMeasure(distfunc)
    kmeansmodel = kmeans.fit(dataFrame)
    t_cost = time.time() - t_begin
    print('time cost = %.4f' % t_cost)

    # Make predictions
    predictions = kmeansmodel.transform(dataFrame)
    labels_cluster = np.array(predictions.select('prediction').toPandas().values)

    # Evaluation: Silhouette score
    # evaluator = ClusteringEvaluator().setFeaturesCol('features').setPredictionCol('prediction')
    # silhouette = evaluator.evaluate(predictions)
    # print("Silhouette with squared euclidean distance = " + str(silhouette))

    # Evaluation: CH score
    CH = metrics.calinski_harabaz_score(labels_cluster, labels_gt)
    print("KMeans clustering CH score = %.4f " % CH)

    # Evaluation: Silhouette score
    SS = metrics.silhouette_score(labels_cluster, labels_gt, metric='euclidean')
    print("KMeans clustering Silhouette score = %.4f " % SS)

    # Evaluation: SSE
    SSE = kmeansmodel.computeCost(dataFrame)
    print('KMeans clustering sum of squared distances = %.4f ' % SSE)

    # Evaluation: Purity
    # print(type(labels_cluster))
    # print(np.shape(labels_cluster))
    # print(type(labels_gt))
    # print(np.shape(labels_gt))
    Purity = evaluateKMeans(labels_cluster.squeeze(), labels_gt)


    f = open('./log.txt','a')
    f.write('\n')
    f.write(time.asctime( time.localtime(time.time()) ))
    f.write('\n SSE = %f \n Prec = %f \n CH = %f \n SS = %f'% (SSE, Purity, CH, SS))
    f.write('\n')
