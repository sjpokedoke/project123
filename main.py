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
import PIL.ImageOps
import os, ssl, time

if (not ox.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "_create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)
print(pd.Series(y).value_counts())
classess = ['A', 'B', 'C', 'D', 'E' 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
xtrainscaled = xtrain/255.0
xtestscaled = xtest/255.0

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(xtrainscaled, ytrain)

ypred = clf.predict(xtestscaled)

accuracy = accuracy_score(ytest, ypred)
print("Accuracy is: ", accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        upperleft = (int(width/2-56), int(height/2-56))
        bottomright = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upperleft, bottomright, (0, 255, 0), 2)

        roi = gray[upperleft[1]:bottomright[1], upperleft[0]:bottomright[0]]
        impil = Image.fromarray(roi)

        imagebw = impil.convert('L')
        imagebwresized = imagebw.resize((28, 28), Image.ANTIALIAS)
        imagebwresizedinverted = PIL.ImageOps.invet(imagebwresized)
       
        pixelfilter = 20
        minpixel = np.percentile(imagebwresizedinverted, pixelfilter)
        
        imagebwresizedinvertedscaled = np.clip(imagebwresizedinverted-minpixel, 0, 255)
        maxpixel = np.max(imagebwresizedinverted)

        imagebwresizedinvertedscaled = np.asarray(imagebwresizedinvertedscaled)/maxpixel

        testsample = np.array(imagebwresizedinvertedscaled).reshape(1, 784)
        testpred = clf.predict(testsample)
        
        print("Predicted class: ", testpred)

        cv2.imshow("frame", gray)
        if cv2.waitkey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()