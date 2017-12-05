# declare global variables
import os
import cv2

root=os.path.dirname(os.path.abspath(__file__))+'\\'

DataSetPath=root+'testSet\\'
resultSetPath=root+'testResult\\'
GroundTruthPath=DataSetPath+'data\\'
face_cascade = cv2.CascadeClassifier(root+'src\\haar_cascade\\'+'haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier(root+'src\\haar_cascade\\'+'haarcascade_eye.xml')

flag_show=True



