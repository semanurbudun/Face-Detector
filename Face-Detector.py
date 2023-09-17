import cv2
from cv2 import imshow
from random import randrange

 #Ön yüz okuma verileri yükleme(haar cascade algoritması)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Yüz algılamak için bir görüntü seçin
#img = cv2.imread('RDJ.png')
#img = cv2.imread('unlu.jpg')

#Web kamerasından video yakalamak için
webcam = cv2.VideoCapture(0)

###Çerçeve üzerinde sonsuz yinele 
while True:

        #geçerli çerçeveyi oku
        successful_frame_read, frame = webcam.read()

        #Gri tonlamaya dönüştürme
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
         
        #Yüz tespiti
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

        #Diktörgen çizme
        for(x, y, w, h) in face_coordinates:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 6)
        
        cv2.imshow('Yuz Dedektor',frame)
        key =  cv2.waitKey(1 )

        #Q tuşuna basılırsa dur
        if key==81 or key==113:
            break
webcam.release()

print("Code Completed")

"""

#Resimleri gösterme

cv2.imshow('Face Detector', img)
cv2.waitKey()

"""
