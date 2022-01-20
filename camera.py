import cv2
from model import FacialExpressionModel
import numpy as np
import matplotlib.pyplot as plt

facec = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
model = FacialExpressionModel("Facec_model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(0)
while True:
    res, fr = cap.read()
    if not res:
        continue
        
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y+h, x:x+w]
            
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            
        cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(fr, (x, y), (x+w, y+h), (255, 0, 0), 2)
          
    cv2.imshow("Image", fr)
    if cv2.waitKey(10) == ord('q'):
        break

   
       
cap.release()
cv2.destroyAllWindows()
