import cv2
import os
import fnmatch
import src.FileOperation as fo
import src.haar_cascade.haarcascade as hc

import config
import getCenter




image_sets_name=[]
for root, dir, files in os.walk(config.DataSetPath+'.'):
    if root==config.DataSetPath+'.':
        image_sets_name=files

maxy=-1
miny=1000
maxx=-1
minx=1000

for item in image_sets_name:
    print('image',item)
    truth=fo.readGroundTruth(item)
    src_img=cv2.imread(config.DataSetPath+item)

    (x,y)=getCenter.getCorneaCenter(src_img)
    '''
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    [fx,fy,fw,fh]=hc.detectFace(gray)

    img_face=gray[fy:fy+fh,fx:fx+fw]
    truth_in_face=[truth[0]-fx,truth[1]-fy,truth[2]-fx,truth[3]-fy]
    print('truth',truth_in_face)
    eyes=hc.detectEyes(img_face)
    for[ex,ey,ew,eh] in eyes:
        maxy=max(maxy,ey+eh)
        miny=min(miny,ey)
        print(ex/fw,ey/fh,(ex+ew)/fw,(ey+eh)/fh)

    #cv2.imshow(item,img_face)
    #cv2.waitKey(0)
    #cv2.destroyWindow(item)
    '''
    print('')
print(maxy,miny)
