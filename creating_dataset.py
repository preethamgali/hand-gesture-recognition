import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
 
folder_name = 'dataset'
if folder_name not in os.listdir('.'):
    os.mkdir('./'+folder_name)

new_folder = input("enter the folder name:")
os.mkdir('./'+folder_name+'/'+new_folder)

i = 0 
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

    cv2.imshow("you",f)
    c = cv2.waitKey(1) & 0xFF
    if c == ord('q'):
        cv2.destroyAllWindows()
        del(camera)
        choose = input("what to creat other gesture?(y/n):")
        if choose in ['Y','y']:
            new_folder = input("enter the folder name:")
            os.mkdir('./'+folder_name+'/'+new_folder)
            camera = cv2.VideoCapture(0)
            i = 0
        else:
            break
    elif c == ord(' '):
        f = f[50:250,400:600]
        i += 1
        file_name = './'+folder_name+'/'+new_folder+'/'+str(i)+'.jpg'
        cv2.imwrite(file_name,f)
        print(f.shape)


# print(len(peace))
# while cv2.waitKey(1) & 0xFF != ord('q'):
#     plt.imshow(f[1])
