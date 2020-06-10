import os
import csv
import cv2
import copy
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def altlight(img):
    nimg = copy.deepcopy(img)
    nimg = np.int32(nimg)
    rdn = random.randint(0,100)
    dl = [-1,1]
    i = dl[random.randint(0,1)]    
    nimg += rdn*i
    nimg[nimg < 0] = 0
    nimg[nimg > 255] = 255
    
    return np.uint8(nimg)
    

def saltpepper(img,prob):
    output = np.zeros(img.shape,np.uint8)
    ng = 1-prob
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = random.random() # 0 - 1 random variable
            if rdn < prob:
                output[i][j][:] = 0
            elif rdn > ng:
                output[i][j][:] = 255
            else:
                output[i][j][:] = img[i][j][:]
                
    return output


def rotate(img,angle,file,c,bbox,label,y,scale=1.0):
    nimg = copy.deepcopy(img)
    h,w,d = ([i for i in img.shape])
    bbox = np.hstack((np.hstack((bbox['x1'],bbox['y1'],bbox['x2'],bbox['y2']))))
    
    cX, cY = (w//2,h//2)
    M = cv2.getRotationMatrix2D((cX,cY),angle,scale)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    
    nimg = cv2.warpAffine(nimg,M,(nW,nH))
    
    corners = getcorners(bbox)
    corners = rotatebbox(corners,angle,cX,cY,h,w)
    newbbox = (get_enclosing_box(corners))[0]
    
    scale_factor_x = nimg.shape[0]/w
    scale_factor_y = nimg.shape[1]/h
    newbbox /= [scale_factor_x,scale_factor_y,scale_factor_x,scale_factor_y]
    
    nimg = cv2.resize(nimg,(w,h))
    st = saltpepper(nimg,0.02)
    alt = altlight(nimg)
    
    # cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),2)
    #cv2.rectangle(nimg,(int(newbbox[0]),int(newbbox[1])),(int(newbbox[2]),int(newbbox[3])),(0,255,0),2)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    #cv2.imshow('image',nimg)
    #cv2.waitKey(0)
    
    new_data = pd.DataFrame(columns=y.columns)
    new_data['id'] = ['img_'+str(c)+'.png','img_'+str(c+1)+'.png','img_'+str(c+2)+'.png']
    new_data['class'] = label
    
    new_data['x1'] = newbbox[0]
    new_data['y1'] = newbbox[1]
    new_data['x2'] = newbbox[2]
    new_data['y2'] = newbbox[3]
    
    cv2.imwrite(os.path.join(file,'img_'+str(c)+'.png'),nimg)
    cv2.imwrite(os.path.join(file,'img_'+str(c+1)+'.png'),st)
    cv2.imwrite(os.path.join(file,'img_'+str(c+2)+'.png'),alt)
    
    return new_data
    

def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def rotatebbox(corners,angle,cx,cy,h,w):
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1),dtype=type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    return calculated.reshape(-1,8)

def getcorners(bboxes):
    width = (bboxes[2] - bboxes[0]).reshape(-1,1)
    height = (bboxes[3] - bboxes[1]).reshape(-1,1)
    
    x1 = bboxes[0].reshape(-1,1)
    y1 = bboxes[1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[2].reshape(-1,1)
    y4 = bboxes[3].reshape(-1,1)
    
    return np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

# Open csv bounding box file
with open('bbox.csv','r') as csvfile:
    reader = pd.read_csv(csvfile)    
    csvfile.close()

# Data augmentation
counter = 0
y = pd.DataFrame()
for dirname, dirs, filenames in os.walk('Dataset/'):
    count = len(filenames)
    y = y.append(reader.iloc[counter:counter+len(filenames)])

    for idx in range(0,len(filenames)):
        label = os.path.basename(dirname)
        ImgName = os.path.join(dirname,"img_"+str(idx)+".png")
        Img = cv2.imread(ImgName)
                
        clk = [-1,1]                        
        i = random.randint(40, 360)
        newlabels = rotate(Img,i*clk[random.randint(0,1)],os.path.join(dirname), \
                        count+idx,reader.iloc[counter+idx][2:6],label,y,1.0)
            
        y = y.append(newlabels,ignore_index=True)
        
        count += 2
        
    counter += len(filenames)

y.to_csv('train.csv')


## finish
