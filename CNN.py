###############################################                    CNN in KERAS         #######################################################

import numpy as np
import tensorflow as tf
from  tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from data_prep import train_images, train_labels, categories
import cv2
import time
import os


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(200,200,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

def train():
    try:
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        model.fit(history = model.fit(train_images, train_labels, epochs=10000, validation_data=(train_images, train_labels)))
    except:
        pass
    model.save_weights('./checkpoints/my_checkpoint')

def useModel():

    model.load_weights('./checkpoints/my_checkpoint')

    print('\n\n')
    print(' '*100)

    output = {}
    output_img = {}


    for i in range(len(categories)):
        for emojies in os.listdir("./handgesture"):
            if categories[i] in emojies:
                output_img[i] = emojies
        output[i] = categories[i]
    

    camera = cv2.VideoCapture(0)


    while True:
        r,f = camera.read()
        f = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        line_t = 5
        f = cv2.flip(f,1)
        f[50:250,400-line_t:400] = np.full((200,line_t),0)
        f[50:250,600:600+line_t] = np.full((200,line_t),0)
        f[50-line_t:50,400:600] = np.full((line_t,200),0)
        f[250:250+line_t,400:600] = np.full((line_t,200),0)

        cv2.imshow("Webcam",f)
        f = f[50:250,400:600]
        f.shape = (f.shape[0],f.shape[1],1)

        res = np.argmax(model.predict([[f]]))
        emojie = cv2.resize(cv2.imread('./handgesture/'+output_img[res]),(100,100))
        cv2.imshow("Emojie idemntified",emojie)

        # print('                                     ',output[res],end = '\r')

        c = cv2.waitKey(1) & 0xFF
        if c == ord('q'):
            cv2.destroyAllWindows()
            del(camera)
            break
    

# train()
useModel()


