import numpy as np
import os
import cv2

categories = os.listdir("./dataset")
print(categories)
y  = {}
train_images, train_labels = [],[]

for i in range(len(categories)):
    y[categories[i]] = i

for category in categories:
    images = os.listdir("./dataset/"+category)
    for img in images:
        img = cv2.imread("./dataset/"+category+"/"+img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img.shape = (img.shape[0],img.shape[1],1)
        train_images.append(img)
        train_labels.append(y[category])

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)