from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import argparse
import utils
import numpy as np

#Given an image, compute a K-Means cluster

image = cv2.imread(r"C:\Users\Cullen\Pictures\Camera Roll\PicOfMe.PNG")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
"""with np.printoptions(threshold=np.inf):
    print(image)"""

"""plt.figure()
plt.axis("off")
plt.imshow(image)
plt.show()"""

#reshape matrix
image = image.reshape((image.shape[1]*image.shape[0],3))

#Use the K-Means algorithm
kmeans = KMeans(n_clusters=5)
s = kmeans.fit(image)

#These are the values that are assigned to each pixel
labels = kmeans.labels_
with np.printoptions(threshold=np.inf):
    print(labels)
#print(labels)
labels = list(labels)

#K arrays of 3 values. Each value represents an RGB value respectively
centroid = kmeans.cluster_centers_
print(centroid)

"""percent = []
for i in range(len(centroid)):
    j = labels.count(i)
    j = j / (len(labels))
    percent.append(j)
print(percent)

plt.pie(percent, colors=np.array(centroid/255), labels = np.arange(len(centroid)))
plt.show()
"""
