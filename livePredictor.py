import numpy as np
import cv2
import glob
import csv
import dlib
import math
from sklearn.externals import joblib


#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file

############################OPENCV CAPTURE VIDEO FRAME BY FRAME#####################################
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised





## detecting face using haar cascades
def detect_live_face(frame):
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale    

#Detect face using 4 different classifiers
    face = faceDet.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face2 = faceDet2.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face3 = faceDet3.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face4 = faceDet4.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)


    if len(face) == 1:
        facefeatures = face
    elif len(face2) == 1:
        facefeatures == face2
    elif len(face3) == 1:
        facefeatures = face3
    elif len(face4) == 1:
        facefeatures = face4
    else:
        facefeatures = ""
    
    #Cut and save face
    for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
        frame = frame[y:y+h, x:x+w] #Cut the frame to size
        
        try:
            out = cv2.resize(frame, (350, 350)) #Resize face so all images have same size
            return out
        except:
           return None

## load the pretrained classifier
clf = joblib.load('model2.pkl')
emotions = ["anger", "disgust", "happy", "neutral", "surprise"] #Emotion list
# video_capture = cv2.VideoCapture(0) #Webcam object
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
faceDet = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt_tree.xml")
i=0
message=""
while True:
    ret, frame = video_capture.read() 
    height, width, channels = frame.shape
    if i == 1:
    	i=-6
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    	clahe_image = clahe.apply(gray)
    	landmarks=get_landmarks(clahe_image)
    	if not landmarks == "error":
    		emo=clf.predict(landmarks)
    		message=emotions[emo]
        else:
            message="Something new!"
    i=i+1
    cv2.putText(frame, message ,(width/2-30,20), font, 1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("image", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
