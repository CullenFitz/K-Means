from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Main function
def KMeans(k):

    #image to be used
    image = cv2.imread(r"C:\Users\Cullen\Pictures\Camera Roll\PicOfMe.PNG")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape image matrix
    image = image.reshape((image.shape[1] * image.shape[0], 3))

    