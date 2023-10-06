import cv2 as cv
import glob
import os
import sys
import pandas as pd

path = str(sys.argv[1])
print(path) 
def RGB(path):
    all_files=glob.glob(os.path.join(path, '*.PNG'))
    Color=[]
    for files in all_files:
        img=cv.imread(files)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        R=[]
        G=[]
        B=[]
        for height in range(img.shape[0]):
            for width in range(img.shape[1]):
                r,g,b = img[height, width]
                R.append(r)
                G.append(g)
                B.append(b)
        
        AverageB=sum(B)/len(B)
        AverageG=sum(G)/len(G)
        AverageR=sum(R)/len(R)
        Image=os.path.basename(files).split('.')[0]

        Color.append(
            {'file':Image,
             'Red':AverageR,
             'Green':AverageG,
             'Blue':AverageB
            }
        )
    RGB_values=pd.DataFrame(Color)
    print(RGB_values)

RGB(path)