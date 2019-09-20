
#Motion Detection using OpenCV

import cv2 

import time

firstframe=None

video=cv2.VideoCapture(0)

while True:

	check,frame = video.read()

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)                                                   

	gray = cv2.GaussianBlur(gray,(21,21),0)                                                         
	if firstframe is None:

		firstframe = gray
		continue

	deltaframe = cv2.absdiff(firstframe,gray)                                                       

	threshframe = cv2.threshold(deltaframe,30,255,cv2.THRESH_BINARY)[1]

	threshframe = cv2.dilate(threshframe,None,iterations=2)

	(cnts,_) = cv2.findContours(threshframe.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	for c in cnts:

		if cv2.contourArea(c) < 10000:
			continue

		(x,y,w,h) = cv2.boundingRect(c)

		cv2.rectangle(frame,(x,y),(x+w,y+h),(125,125,0),3)

	#cv2.imshow("capturing",gray)
	cv2.imshow("delta",deltaframe)
	cv2.imshow("threshold frame",threshframe)
	cv2.imshow("color frame",frame)

	key = cv2.waitKey(1)
	#print(gray)
	if key == ord('q'):
		break

video.release()

cv2.destroyAllWindows