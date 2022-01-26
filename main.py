from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np


# Main function
def kmeans(k, blk):
    # image to be used
    if blk:
        image = cv2.imread(r"/Users/cullenfitzgerald/Downloads/PicOfMe.PNG")
    else:
        image = cv2.imread(r"/Users/cullenfitzgerald/Downloads/normalPic.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image matrix
    image = image.reshape((image.shape[1] * image.shape[0], 3))

    finArr = []
    itr = 10
    for i in range(itr):
        kmeans = KMeans(n_clusters=k)
        s = kmeans.fit(image)

        # These are the values that are assigned to each pixel. Labels is the array that we want to average
        labels = kmeans.labels_
        finArr.append(labels)

    for i in range(k):
        finMat = []
        for j in range(len(finArr[0])):
            temp = []
            for n in range(itr):
                temp.append(finArr[n][j])
            tot = temp.count(i)
            tot = tot / itr
            finMat.append(tot)
        finMat = np.reshape(finMat, (-1, 2))
        heat_map = sns.heatmap(finMat,cmap='gist_ncar')
        heat_map.set_title("Cluster " + str(i))
        plt.xticks([])
        plt.yticks([])
        plt.show()

kmeans(3, blk=False)
