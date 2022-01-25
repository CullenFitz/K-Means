from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Main function
def kmeans(k):
    # image to be used
    image = cv2.imread(r"/Users/cullenfitzgerald/Downloads/PicOfMe.PNG")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image matrix
    image = image.reshape((image.shape[1] * image.shape[0], 3))

    finArr = []
    for i in range(10):
        kmeans = KMeans(n_clusters=k)
        s = kmeans.fit(image)

        # These are the values that are assigned to each pixel. Labels is the array that we want to average
        labels = kmeans.labels_
        #labels = list(labels)
        finArr.append(labels)
        #print(finArr[i])

    plot = []
    for i in range(k):
        finMat = []
        for j in range(len(finArr[0])):
            temp = []
            for n in range(10):
                temp.append(finArr[n][j])
            tot = temp.count(i)
            tot = tot / 10
            finMat.append(tot)
        finMat = np.reshape(finMat, (-1, 2))

        heat_map = sns.heatmap(finMat, vmin= 0, vmax = 1, cmap='gist_ncar')
        plt.show()

kmeans(3)
