import cv2
import os
import fnmatch
import src.FileOperation as fo
import src.haar_cascade.haarcascade as hc

import config
import getCenter

filter_method_list=['mean','median','morph','none']
filter_size_list=[3,5,7,11,21]
edge_method_list=['canny','sobel','laplacian']
threshold1_list=range(10,60,5)
threshold2_list=range(30,150,10)
hough_dp_list=[1,1.5,2,2.5,3]

morph_kernel_rate_list=range(2,10,1)
threshold_method_list=['otsu','peak']
flag_precise=[True,False]

########################################################################
#Experiment Main
########################################################################
image_sets_name=[]
for root, dir, files in os.walk(config.DataSetPath+'.'):
    if root==config.DataSetPath+'.':
        image_sets_name=files

def ifInclude(circle,circle_truth):
    if circle[2]<circle_truth[2]:
        return 0
    dis=(circle[0]-circle_truth[0])**2+(circle[1]-circle_truth[1])**2
    boundary=(circle[2]-circle_truth[2])**2
    return 1*(dis<boundary)
def calculateDise(circle,circle_truth):
    return (circle[0]-circle_truth[0])**2+(circle[1]-circle_truth[1])**2

sum=0
for item in image_sets_name:
    print('image',item)
    truth=fo.readGroundTruth(item)
    src_img=cv2.imread(config.DataSetPath+item)
    point=getCenter.getCorneaCenter(src_img)
    sum+=calculateDise(point,truth)
    print('loss',calculateDise(point,truth))
    print('')
result=sum/len(image_sets_name)
print(result)

