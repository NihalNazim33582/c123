import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import time
import ssl
import PIL.ImageOps

# setting https to fetch data

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)
# print(pd.Series(y).value_counts())
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nClasses = len(classes)
# print(nClasses)

smaple_per_class = 5
figure = plt.figure(figsize=(10, 10))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=9, train_size=7500, test_size=2500)
# print((X_train[25]))

X_train_scale = X_train/255.0
X_test_scale = X_test/255.0
# print(X_train_scale[25])

model = LogisticRegression(
    solver='saga', multi_class='multinomial').fit(X_train_scale, y_train)

y_predict = model.predict(X_test_scale)
A = accuracy_score(y_predict, y_test)
print(A)

capture=cv2.VideoCapture(0)
while True:
    try:
        rect,frame=capture.read()
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        #  Drawing rectangle at the center fo the screen

        height,width=gray.shape

        upperLeft=(int(width/2-250),int(height/2-250))
        bottomRight=(int(width/2+250),int(height/2+250))

        cv2.rectangle(gray, upperLeft, bottomRight, (0,0,255),3)

        IntrestingRegion=gray[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]

        #Convertign cv2 to python image libary

        Image_py=Image.fromarray(IntrestingRegion)

        #Converting Region of intrest to gray scale using L mode
        image_bw=Image_py.convert('L')
        image_bw_resize=image_bw.resize((28,28),Image.ANTIALIAS)

        # Inverting image using imageop
        InvertImage=PIL.ImageOps.invert(image_bw_resize)

        percentVal=50
        minPixels=np.percentile(InvertImage,percentVal)
        InvertImageScale=np.clip(InvertImage-minPixels,0,255)

        maxPixels=np.max(InvertImage)
        InvertImageScale=np.asarray(InvertImage.scale)/maxPixels

        Test_sample=np.array(InvertImageScale).reshape(1, 784)

        test_predict=model.predict(test_sample)
        print(test_predict)

        cv2.imshow("Digt Two", gray)

        if cv2.waitKey(5)== ord("a"):
            break

    except Exception as e: 
        pass
capture.release()
cv2.destroyAllWindows()
