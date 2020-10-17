#!/usr/bin/python

import cv2
from matplotlib import pyplot as plt

def myimshow(img, time=None):
    
    cv2.imshow("image",img)
    
    if time is None:
        cv2.waitKey(0)    
    else:
        cv2.waitKey(time)
        
    cv2.destroyAllWindows()


def myimplot(img):
    
    if len(img.shape)>2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, 'gray')
        
    
    plt.show()
    
