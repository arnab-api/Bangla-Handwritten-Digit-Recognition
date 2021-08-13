import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import imutils
import random


def display(img, img2):
    cv2.namedWindow('img1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 600,700)
    cv2.imshow("img1", img)
    
    cv2.namedWindow('img2',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 600,700)
    cv2.imshow("img2", img2)
    
    cv2.waitKey(0)
    
    return

def showImage(img):
    plt.imshow(img,cmap='gray',interpolation='bicubic')
    plt.show()
    
    return

def finalPreprocess(original_image):
    
    (thresh, box_bw) = cv2.threshold(original_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    noise_removal = 'YES'
    
    box_bw_border_free=box_bw
 
    if noise_removal.startswith('Y') or noise_removal.startswith('y'):
        box_bw_border_free=cv2.medianBlur(box_bw_border_free,3)
        
    H1 = np.size(box_bw_border_free, 0)
    W1 = np.size(box_bw_border_free, 1)
    
    (thresh, In_bw) = cv2.threshold(box_bw_border_free,128, 255, cv2.THRESH_BINARY)
    inverted_In_bw=np.invert(In_bw) 
    (i,j)=np.nonzero(inverted_In_bw) 
    if np.size(i)!=0:
        Out_cropped = box_bw_border_free[np.min(i):np.max(i),np.min(j):np.max(j)] 
    else: 
         Out_cropped = box_bw_border_free
         
         
         
    height = 50
    width = 50
    
    H = np.size(Out_cropped, 0)
    W = np.size(Out_cropped, 1)
    
    if H==0 or W==0:
        Out_cropped = box_bw_border_free
    else:
        if (W/H)<0.4 or (H/W)<0.4 or (H/H1)<0.25 or (W/W1)<0.25:
            border_width = 0.1
            box_bw_thinned_bordered = cv2.copyMakeBorder(Out_cropped, int(height*border_width), int(height*border_width), int(width*border_width), int(width*border_width), cv2.BORDER_CONSTANT, value=255)  
            struc_element = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
            Out_cropped=cv2.erode(box_bw_thinned_bordered,struc_element)    
    
    
    Ithin_resized=cv2.resize(Out_cropped,(width,height),None,0,0,cv2.INTER_LANCZOS4) 

    (thresh, Ithin_resized_thresh) = cv2.threshold(Ithin_resized,200, 255, cv2.THRESH_BINARY)
  
    box_bw_thinned=(Ithin_resized_thresh)
    border_width = 0.3
    box_bw_thinned_bordered = cv2.copyMakeBorder(box_bw_thinned, int(height*border_width), int(height*border_width), int(width*border_width), int(width*border_width), cv2.BORDER_CONSTANT, value=255)  

    struc_element = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    Output=cv2.erode(box_bw_thinned_bordered,struc_element)
    
    return Output



def PreProcess_1(img, crop=True, m = 5):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = cv2.medianBlur(img,3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 10)

    if len(img[img==255]) > len(img[img==0]):
        img = cv2.bitwise_not(img)

    if crop:      
        _,contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if(len(contours)==0):
            return img
        
        cMax = max(contours, key=cv2.contourArea) 

        x,y,w,h = cv2.boundingRect(cMax)

        x1 = (x-m) if (x-m)>0 else x
        y1 = (y-m) if (y-m)>0 else y
        
        x2 = (x+w+m) if (x+w+m)<img.shape[1] else (x+w)
        y2 = (y+h+m) if (y+h+m)<img.shape[0] else (y+h)

        img = img[ y1:y2, x1:x2 ]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    
    img = 255-img
    img = finalPreprocess(img)
    img = 255-img
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    
    return img


def PreProcess(img, name, crop=True, m = 5):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height = np.size(img, 0)
    width = np.size(img, 1)
    
#    for i in range(height):
#        for j in range(width):
#            if(img[i, j]==0):
#                img[i, j]=235
                
                
    
    if(name[0]=='e'):
        img = 255-img
        img = finalPreprocess(img)
        img = 255-img
        return img
    
    if(name[0]=='f'):
        height = np.size(img, 0)
        width = np.size(img, 1)
        
        name=name[5]+name[6]+name[7]
        
        val = 255
        
        if(name=="014" or name=="060" or name=="122" or name=="148" or name=="169"):
            val=190
        if(name=="170" or name=="199" or name=="265" or name=="307" or name=="323"):
            val=190
        if(name=="328" or name=="347" or name=="427" or name=="432" or name=="441" or name=="487" or name=="493"):
            val=190
          
        if(val==190):
            
            for i in range(height):
                for j in range(width):
                    if(img[i, j]>=250):
                        img[i, j]=val
                

    img = cv2.medianBlur(img,3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 10)

    if len(img[img==255]) > len(img[img==0]):
        img = cv2.bitwise_not(img)

    if crop:      
        _,contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if(len(contours)==0):
            return img
        
        cMax = max(contours, key=cv2.contourArea) 

        x,y,w,h = cv2.boundingRect(cMax)

        x1 = (x-m) if (x-m)>0 else x
        y1 = (y-m) if (y-m)>0 else y
        
        x2 = (x+w+m) if (x+w+m)<img.shape[1] else (x+w)
        y2 = (y+h+m) if (y+h+m)<img.shape[0] else (y+h)

        img = img[ y1:y2, x1:x2 ]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    
    img = 255-img
    img = finalPreprocess(img)
    img = 255-img
    
    return img


def getArr(img):
    
    height = np.size(img, 0)
    width = np.size(img, 1) 
    
    arr = []
    
    i=0
    
    while i<height:
    
        j=0
        
        while j<width:
            
            px = img[i, j]
            
            arr.append(px/255)
            
            j = j+1
        
        i = i+1
    
    return arr


f1 = open('Training_All_Rotated_50_50.txt','w')

cnt=0

for xx in range(5):
    
    ss = chr(xx+97)
    
    for path in glob.glob("Datasets/training-"+ss+"/*"):
        
        img = cv2.imread(path)
        
        rand1 = random.randint(1,45)
        rand2 = random.randint(1, 45)
        rand2 = -rand2
        
        sz = int(len(path))
        name = path[sz-12:sz]
        
        
        img = PreProcess(img, name)
        
        img = cv2.resize(img, (50, 50))
        img2 = imutils.rotate_bound(img, rand1)
        img2 = cv2.resize(img2, (50, 50))
        img3 = imutils.rotate_bound(img, rand2)
        img3 = cv2.resize(img3, (50, 50))
        
        arr = getArr(img)
        arr2 = getArr(img2)
        arr3 = getArr(img3)
#        
#        showImage(img)
#        showImage(img2)
#        showImage(img3)
        
     
        name = name[2:len(name)]
#        print(name, len(arr))
        
        for i in range(len(arr)):
            
            f1.write(str(arr[i])+', ')
        
        f1.write(str(name)+'\n')
#        
        for i in range(len(arr2)):
            
            f1.write(str(arr2[i])+', ')
        
        f1.write(str(name)+'\n')
        
        for i in range(len(arr3)):
            
            f1.write(str(arr3[i])+', ')
        
        f1.write(str(name)+'\n')
#       
        if(cnt%100==0):
            print(cnt, name)
        
        cnt+=1
        
f1.close()
