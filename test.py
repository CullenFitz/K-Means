import numpy
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Main function
def kmeans(k):

    #image to be used
    image = cv2.imread(r"C:\Users\Cullen\Pictures\Camera Roll\PicOfMe.PNG")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image matrix
    image = image.reshape((image.shape[1] * image.shape[0], 3))

    finArr = []

    # Use the K-Means algorithm
    for i in range(10):
        kmeans = KMeans(n_clusters=k)
        s = kmeans.fit(image)

        # These are the values that are assigned to each pixel. Labels is the array that we want to average
        labels = kmeans.labels_
        labels = list(labels)   #needed for counting values

        #add this new array to finArr
        finArr.append(labels)

    for i in finArr:
        finMat = numpy.column_stack((i, i+i))
    print(finMat)

    #keep track of percentages for each iteration
    percents = []
    for i in range(k):
        totOcc = 0
        totDen = 0
        for j in finArr:
            occ = j.count(i)
            totOcc = totOcc + occ
            totDen = totDen + len(j)
        total = totOcc / totDen
        percents.append(total)
    print(percents)

    #find percentages of clusters using matrices from finArr
kmeans(3)