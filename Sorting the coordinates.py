#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import string
from sklearn.cluster import DBSCAN
import numpy as np
import statistics


# In[2]:


f = open("res_test1.txt", "r")
text = f.read()


# In[3]:


test_text = ""
line = text.splitlines()
for ind, i in enumerate(line):
    one_line = i.split(",")
    x1, y1, x2, y2, x3, y3, x4, y4 = one_line[0], one_line[1], one_line[2], one_line[3], one_line[4], one_line[5], one_line[6], one_line[7]
    xc = (int(x1) + int(x2) + int(x3) + int(x4)) // 4
    yc = (int(y1) + int(y2) + int(y3) + int(y4)) // 4
    test_text = test_text + i  + "," + str(ind) + "," + str(xc) + "," + str(yc) + "\n"


# In[4]:


arr = test_text.split("\n")


# In[5]:


res_arr = []
for j in arr:
    try:
        res_arr.append(tuple(map(int, j.split(','))))
    except:
        continue


# In[6]:


import pandas as pd
df = pd.DataFrame(res_arr)


# In[15]:


db = DBSCAN(eps = 10, min_samples=1)


# In[16]:


df1 = pd.DataFrame(df[10])
df1['y'] = 1


# In[17]:


db.fit(df1)


# In[18]:


df['class'] = db.labels_ 


# In[19]:


df.sort_values(by='class')


# In[20]:


df['class']


# In[21]:


#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])


# In[23]:


for k in range(len(df)):
    arr = np.array(df.loc[df['class'] == k])
    arr = arr[arr[:, 9].argsort()]
    for i in range(len(arr)):
        y1 = arr[i][1]
        x1 = arr[i][0]
        x2 = arr[i][2]
        y2 = arr[i][3]
        y3 = arr[i][5]
        h = int(y3) - int(y2)
        w = int(x2) - int(x1)
        img = cv2.imread("res_test1.jpg")
        img = img[int(y1) : int(y1) + h, int(x1) : int(x1) + w]
        try:
            plt.figure()
            plt.imshow(img)
        except:
            continue
        #cv2.imwrite(str(str(x1) + str(y1) + ".jpg") , img)
        #img = cv2.filter2D(img, -1, kernel)
        #text_in_image = pytesseract.image_to_string(img)


# In[ ]:




