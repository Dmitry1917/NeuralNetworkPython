import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

numberOfMethods = 8
numberOfExamples = 7
plt.figure(figsize = (5 * numberOfExamples, 5 * numberOfMethods), num = 1, clear = True)

def plot(x, y, data, numberOfColumns, column, info, equalAxis = False):

    plt.subplot(numberOfMethods, numberOfColumns, column + 1)
    kmeans = KMeans(n_clusters = 3)
    kmeans.fit(data)

    if equalAxis:
        plt.axis('equal')
    plt.title(f'KMeans 3 {info}')
    plt.scatter(x, y, c = kmeans.labels_)
    
    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns + column + 1)
    dbscan = DBSCAN()
    dbscan.fit(data)

    if equalAxis:
        plt.axis('equal')   
    dbScanNumberOfClusters = len(set(dbscan.labels_))
    plt.title(f'DBSCAN {dbScanNumberOfClusters} {info}')
    plt.scatter(x, y, c = dbscan.labels_)
    
    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns * 2 + column + 1)
    hdbscan = HDBSCAN()
    hdbscan.fit(data)

    if equalAxis:
        plt.axis('equal')   
    hdbScanNumberOfClusters = len(set(hdbscan.labels_))
    plt.title(f'HDBSCAN {hdbScanNumberOfClusters} {info}')
    plt.scatter(x, y, c = hdbscan.labels_)
    
    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns * 3 + column + 1)
    optics = OPTICS()
    optics.fit(data)

    if equalAxis:
        plt.axis('equal')   
    opticsNumberOfClusters = len(set(optics.labels_))
    plt.title(f'OPTICS {opticsNumberOfClusters} {info}')
    plt.scatter(x, y, c = optics.labels_)


    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns * 4 + column + 1)
    spectral = SpectralClustering(n_clusters = 3)
    spectral.fit(data)
 
    if equalAxis:
        plt.axis('equal')   
    plt.title(f'Spectral 3 {info}')
    plt.scatter(x, y, c = spectral.labels_)

    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns * 5 + column + 1)
    gm = GaussianMixture(n_components = 3)
    labels = gm.fit_predict(data)
 
    if equalAxis:
        plt.axis('equal')   
    gmNumberOfClusters = len(set(labels))
    plt.title(f'GM {gmNumberOfClusters} {info}')
    plt.scatter(x, y, c = labels)

    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns * 6 + column + 1)
    gm = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
    labels = gm.fit_predict(data)
 
    if equalAxis:
        plt.axis('equal')   
    plt.title(f'AC Ward 3 {info}')
    plt.scatter(x, y, c = labels)

    plt.subplot(numberOfMethods, numberOfColumns, numberOfColumns * 7 + column + 1)
    gm = AgglomerativeClustering(n_clusters = 3, linkage = 'single')
    labels = gm.fit_predict(data)
 
    if equalAxis:
        plt.axis('equal')   
    plt.title(f'AC Single 3 {info}')
    plt.scatter(x, y, c = labels)




size = 100

dataX1 = np.random.normal(loc = 5.0, scale = 2, size = size)
dataY1 = np.random.normal(loc = 3.0, scale = 1, size = size)

dataX2 = np.random.normal(loc = 20.0, scale = 2, size = size)
dataY2 = np.random.normal(loc = 3.0, scale = 1, size = size)

dataX3 = np.random.normal(loc = 12.0, scale = 2, size = size)
dataY3 = np.random.normal(loc = 7.0, scale = 1, size = size)

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 0, 'normal')


dataX1 = np.random.uniform(3, 7, size = size)
dataY1 = np.random.uniform(2, 4, size = size)

dataX2 = np.random.uniform(14, 18, size = size)
dataY2 = np.random.uniform(2, 4, size = size)

dataX3 = np.random.uniform(6.5, 10.5, size = size)
dataY3 = np.random.uniform(4.5, 6.5, size = size)

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 1, 'uniform')


dataX1 = np.random.normal(loc = 5.0, scale = 2, size = size)
dataY1 = np.random.normal(loc = 3.0, scale = 1, size = size)

dataX2 = np.random.uniform(14, 18, size = size)
dataY2 = np.random.uniform(2, 4, size = size)

dataX3 = np.random.normal(loc = 12.0, scale = 2, size = size)
dataY3 = np.random.normal(loc = 7.0, scale = 1, size = size)

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 2, 'normal and uniform')


steps = np.linspace(0, 2 * np.pi, size)
dataX1 = np.array([5 + 3 * np.cos(angle) for angle in steps])
dataY1 = np.array([3 + 3 * np.sin(angle) for angle in steps])

dataX2 = np.array([5 + 2 * np.cos(angle) for angle in steps])
dataY2 = np.array([3 + 2 * np.sin(angle) for angle in steps])

dataX3 = np.array([5 + 1 * np.cos(angle) for angle in steps])
dataY3 = np.array([3 + 1 * np.sin(angle) for angle in steps])

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 3, 'circles', equalAxis = True)


dataX1 = np.array([5 + 6 * np.cos(angle) for angle in steps])
dataY1 = np.array([3 + 3 * np.sin(angle) for angle in steps])

dataX2 = np.array([5 + 4 * np.cos(angle) for angle in steps])
dataY2 = np.array([3 + 2 * np.sin(angle) for angle in steps])

dataX3 = np.array([5 + 2 * np.cos(angle) for angle in steps])
dataY3 = np.array([3 + 1 * np.sin(angle) for angle in steps])

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 4, 'ellipses', equalAxis = True)


steps = np.linspace(-np.pi / 2, np.pi / 2, size)
dataX1 = np.array([5 + 6 * np.cos(angle) for angle in steps])
dataY1 = np.array([3 + 3 * np.sin(angle) for angle in steps])

dataX2 = np.array([5 + 4 * np.cos(angle) for angle in steps])
dataY2 = np.array([3 + 2 * np.sin(angle) for angle in steps])

steps = np.linspace(np.pi / 2, 3 * np.pi / 2, size)
dataX3 = np.array([6 + 4 * np.cos(angle) for angle in steps])
dataY3 = np.array([5 + 2 * np.sin(angle) for angle in steps])

x = np.concatenate((dataX1, dataX2, dataX3))
y = np.concatenate((dataY1, dataY2, dataY3))

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 5, 'arcs', equalAxis = True)


x = np.random.uniform(3, 7, size = 3 * size)
y = np.random.uniform(2, 4, size = 3 * size)

data = list(zip(x, y))

plot(x, y, data, numberOfExamples, 6, 'uniform no clusters')

#Elbow method. Look there inertias start to decrease too slow - this is real (or close to) cluster number.
#inertias = []
#for i in range(1, 11):
#    kmeans = KMeans(n_clusters = i)
#    kmeans.fit(data)
#    inertias.append(kmeans.inertia_)
#
#plt.plot(range(1, 11), inertias)
#plt.show()


plt.suptitle('Different clustering methods.')
#plt.show()
plt.savefig('Clustering.png', dpi = 300)


