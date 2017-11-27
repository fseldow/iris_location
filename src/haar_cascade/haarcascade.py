import cv2
import config

# classifiers


def detectFace(gray):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face=config.face_cascade.detectMultiScale(gray,1.1,5)
    if len(face)>1:
        print('warning: find more than 1 head!')
    try:
        print('face pos:','x='+str(face[0][0]),'y='+str(face[0][1]),'w='+str(face[0][2]),'h='+str(face[0][3]))
        return face[0]
    except:
        print('no head found')
        return [0,0,gray.shape[0],gray.shape[1]]

def detectEyes(face_gray):
    face=config.eyes_cascade.detectMultiScale(face_gray)
    return face

