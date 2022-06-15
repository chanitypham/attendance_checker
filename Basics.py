import cv2
import numpy as np
import face_recognition

#import img and convert to rgb
imgChanity = face_recognition.load_image_file('ImagesBasic/phamquynhtrang.jpg')
imgChanity = cv2.cvtColor(imgChanity,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/alicia.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#detect face location, encode face
faceLoc = face_recognition.face_locations(imgChanity)[0]
encodeChanity = face_recognition.face_encodings(imgChanity)[0]
cv2.rectangle(imgChanity,(faceLoc[3],faceLoc[0]), (faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]), (faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#comparisons btw faces, face distances
results = face_recognition.compare_faces([encodeChanity],encodeTest)
faceDis = face_recognition.face_distance([encodeChanity],encodeTest)
#the lower the distance, the better the match

#print result and distance on cmd and pic
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Chanity", imgChanity)
cv2.imshow("Chanity Test", imgTest)
cv2.waitKey(0)