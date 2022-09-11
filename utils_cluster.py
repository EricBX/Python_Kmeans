import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans


def findOptimalK(featureVectorList, minK, maxK, output_path):
    SSE = list() # 样本误差平方和
    for k in range(minK, maxK+1):
        kmeans = KMeans(n_clusters=k, n_jobs=-1)
        kmeans = kmeans.fit(featureVectorList)
        SSE.append(kmeans.inertia_)

    plt.figure()
    plt.title('Using the elbow method to find optimal K value')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.plot(range(minK, maxK+1), SSE, marker="o")
    plt.savefig(os.path.join(output_path,'to_find_optimal_K.png'))
    plt.show()


def visualize_features(n_components, eigenvalues):
    n_col = 5
    n_row = int(n_components / n_col)

    plt.figure()
    for i in list(range(n_row * n_col)):
        offset = 0
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(eigenvalues[i].reshape(28, 28), cmap='jet')
        title_text = 'Eigenvalue ' + str(i + 1)
        plt.title(title_text, size=6.5)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def visualize_kmeans_result(kmeans, features, labels_gt, output_path):
    cents = kmeans.cluster_centers_  # 质心向量（只用到前两维）
    labels = kmeans.labels_  # 样本点被分配到的簇的索引
    sse = kmeans.inertia_ # 所有样本的误差（到最近簇中心的距离）平方之和
    # 画出聚类结果，每一类用一种颜色
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', '#e24fff', '#524C90', '#845868']
    n_clusters = 10
    plt.figure()
    for i in range(n_clusters):
        index = np.nonzero(labels == i)[0]
        x0 = features[index, 0]
        x1 = features[index, 1] # 取特征的前两维做可视化
        y_i = labels_gt[index]
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(int(y_i[j])), color=colors[i],
                     fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(cents[i, 0], cents[i, 1], marker='x', color=colors[i], linewidths=12)
    plt.title("k-means SSE={:.2f}".format(sse))
    xmax = np.max(features[:, 0])
    xmin = np.min(features[:, 0])
    ymax = np.max(features[:, 1])
    ymin = np.min(features[:, 1])
    plt.axis([xmin, xmax, ymin, ymax])
    plt.savefig(os.path.join(output_path, 'kmeans_cluster_visualization' + '.png'))
    plt.show()


def saveResults(labels_pred, data_raw, output_path):
    clusterData = []
    for i in range(10):
        clusterData.append([])
    for i in range(len(labels_pred)):
        clusterData[labels_pred[i]].append(np.reshape(data_raw[i], (28, 28)))

    for l in range(10):
        print("Saving the result of cluster %d" % (l))
        fig, axes = plt.subplots(nrows=10, ncols=10, sharex=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        count = 0
        for i in range(10):
            for j in range(10):
                if count >= len(clusterData[l]):
                    break
                axes[i][j].imshow(clusterData[l][count])
                count += 1
        fig.savefig(os.path.join(output_path , 'kmeans_cluster_' + str(l) + '.png'))


def evaluateKMeans(labels_pred, labels_gt):
    clusterData = []
    for i in range(10):
        clusterData.append([])
    for i in range(len(labels_pred)):
        clusterData[labels_pred[i]].append(labels_gt[i])

    totalPrecision = 0
    for j in range(10):
        labelsClusteredList = []
        for i in range(10):
            labelsClusteredList.append(0)
        for i in range(len(clusterData[j])):
            labelsClusteredList[clusterData[j][i]] += 1 # i: predicted, j: ground-truth
        if sum(labelsClusteredList) == 0:
            clusterPrecision = 0
        else:
            clusterPrecision = max(labelsClusteredList) / sum(labelsClusteredList) * 100.0
        print(labelsClusteredList)
        print('clusterPrecision of %d = %.3f' % (j, clusterPrecision))
        totalPrecision += clusterPrecision
    totalPrecision = totalPrecision / 10
    print("KMeans clustering Precision = %.4f " % totalPrecision)

    return totalPrecision

def findOptimalFeatureDimension(data_std, labels_gt, min_dimension, max_dimension, method, output_path):
    SSE = list() # 样本误差平方和
    Precisions = list() # 聚类准确率
    for feature_dimension in range(min_dimension, max_dimension):
        features = None
        if method == 'PCA':
            pca = PCA(n_components=feature_dimension)
            pca.fit(data_std)
            features = pca.transform(data_std)
        elif method == 'LDA':
            lda = LDA(n_components=feature_dimension)
            lda.fit(data_std, labels_gt)
            features = lda.transform(data_std)
        kmeans = KMeans(n_clusters=10, precompute_distances=True, n_jobs=-1)
        kmeans = kmeans.fit(features)
        SSE.append(kmeans.inertia_)
        Precisions.append(evaluateKMeans(kmeans.labels_, labels_gt))

    plt.figure()
    plt.title('Precisions of k-means with %s of different dimensions' % method)
    plt.xlabel('Number of dimensions')
    plt.ylabel('Precision')
    plt.plot(range(min_dimension, max_dimension), Precisions, marker="o")
    plt.savefig(os.path.join(output_path, 'find_optimal_feature_dimension_of_%s'%method))
    plt.show()