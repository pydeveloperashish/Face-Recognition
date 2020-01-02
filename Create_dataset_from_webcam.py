import cv2
import sys
import os
from os import path

cpt = 0
users = 0;

while True:
	user_images_path = "Give path here/train-images/"+str(users) #Give path to  train-images/ and keep image%04i.jpg as it is in this line. Your images will be stored at train-images/ respective index number folder
	if(path.exists(user_images_path)):
		users += 1
	else:
		os.mkdir(user_images_path)
		break

vidStream = cv2.VideoCapture(0)
while True:
    
    ret, frame = vidStream.read() # read frame and return code.
    
    cv2.imshow("test window", frame) # show image in window
    
    cv2.imwrite(user_images_path+"image%04i.jpg" %cpt, frame)
    cpt += 1
   
    if cv2.waitKey(10)==ord('q'):
        break
        

