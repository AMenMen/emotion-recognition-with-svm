import cv2
import dlib
import sys
from cv2 import WINDOW_NORMAL
import pickle
import numpy as np
import math

detector = dlib.get_frontal_face_detector() #Face detector
#Landmark identifyier. Set the filename to whatever you named the downloaded filename
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks_with_point(image, frame):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #Draw Facial Landmarks with the predictor class
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(17, 68):
            if (i == 27) | (i == 30):
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        #Find both coordinates of centre of gravity
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        #Calculate distance centre <-> other points in both axes
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        #prevent divided by 0 value
        if xlist[11] == xlist[14]:
            angle_nose = 0
        else:
            #point 14 is the tip of the nose, point 11 is the top of the nose brigde
            angle_nose = int(math.atan((ylist[11]-ylist[14])/(xlist[11]-xlist[14]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90

        landmarks = []
        for cx,cy,x,y in zip(xcentral, ycentral, xlist, ylist):
            #Add the coordinates relative to the centre of gravity
            landmarks.append(cx)
            landmarks.append(cy)

            #Get the euclidean distance between each point and the centre point (the vector length)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((y,x))
            dist = np.linalg.norm(coornp-meannp)
            #print(w-xmean)
            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if x == xmean:
                angle_relative = 0
            else:
                angle_relative = (math.atan(float(y-ymean)/(x-xmean))*180/math.pi) - angle_nose
                #print(anglerelative)
            landmarks.append(dist)
            landmarks.append(angle_relative)
        #print('Length x: %d' % len(xcentral))

    if len(detections) < 1:
        #If no face is detected set the data to value "error" to catch detection errors
        landmarks = "error"
    return landmarks

def show_webcam_and_run(model, emotions, window_size=None, window_name='webcam', update_time=10):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    #Set up some required objects
    vc = cv2.VideoCapture(0) #Webcam objects
    if vc.isOpened():
        ret, frame = vc.read()
    else:
        print("webcam not found")
        return

    while ret:
        training_data = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)

        #Get Point and Landmarks
        landmarks_vectorised = get_landmarks_with_point(clahe_image, frame)
        #print(landmarks_vectorised)
        if landmarks_vectorised == "error":
            pass
        else:
            #Predict emotion
            training_data.append(landmarks_vectorised)
            npar_pd = np.array(training_data)
            """prediction_emo_set = model.predict_proba(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo_set = prediction_emo_set[0]
            print(zip(model.classes_, prediction_emo_set))"""
            prediction_emo = model.predict(npar_pd)
            if cv2.__version__ != '3.1.0':
                prediction_emo = prediction_emo[0]
            print(emotions[prediction_emo])

        cv2.imshow(window_name, frame)  #Display the frame
        ret, frame = vc.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):   #Exit program when user press 'q'
            break

def show_image_test(model, emotions):
    training_data = []
    image = cv2.imread('datatest/face4.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    #Get Point and Landmarks
    landmarks_vectorised = get_landmarks_with_point(clahe_image, image)
    #print(landmarks_vectorised)
    if landmarks_vectorised == "error":
        pass
    else:
        #Predict emotion
        training_data.append(landmarks_vectorised)
        npar_pd = np.array(training_data)
        prediction_emo_set = model.predict_proba(npar_pd)
        if cv2.__version__ != '3.1.0':
            prediction_emo_set = prediction_emo_set[0]
        print(zip(model.classes_, prediction_emo_set))
        prediction_emo = model.predict(npar_pd)
        if cv2.__version__ != '3.1.0':
            prediction_emo = prediction_emo[0]
        print(emotions[prediction_emo])

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #emotions = ["happy", "neutral", "sadness","surprise"]
    emotions = ["anger", "disgust", "happy", "neutral", "surprise"]
    #load models
    #joblib.load('models/emotion_detection_model.xml')
    pkl_file = open('models\model1.pkl', 'rb')
    data = pickle.load(pkl_file)
    #data.predict(X[0:1])
    pkl_file.close()


    # use learnt model
    show_image_test(data, emotions)
    #window_name = 'WEBCAM (press q to exit)'
    #show_webcam_and_run(data, emotions, window_size=(800, 600), window_name=window_name, update_time=8)
