import dlib
import cv2
import numpy as np
import glob
import random
import math
import itertools
from sklearn.svm import SVC
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

#Emotion list
emotions = ["anger", "disgust", "happy", "neutral", "surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clf = SVC(C=0.001, kernel='linear', decision_function_shape='ovo', probability=True)   #Set the classifier as a support vector machines with linear kernel

def get_files(emotion):
    files = glob.glob("dataset_combine\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]   #get 80% of image files to be trained
    testing = files[-int(len(files)*0.2):]   #get 20% of image files to be tested
    return training, testing

def get_landmarks(image):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #Draw Facial Landmarks with the predictor class
        shape = model(image, d)
        xlist = []
        ylist = []
        for i in range(17, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        #center points of both axis
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        #Calculate distance between particular points and center point
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

def make_sets():
    training_data = []
    training_label = []
    testing_data = []
    testing_label = []
    for emotion in emotions:
        training, testing = get_files(emotion)
        #Append data to traing and prediction list, and generate labels 0-7
        for item in training:
            #open image
            image = cv2.imread(item)
            #convert to grayscale
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray_img)
            landmarks_vec = get_landmarks(clahe_image)

            if landmarks_vec == "error":
                pass
            else:
                #append image array to trainging data list
                training_data.append(landmarks_vec)
                training_label.append(emotions.index(emotion))

        for item in testing:
            image = cv2.imread(item)
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray_img)
            landmarks_vec = get_landmarks(clahe_image)
            if landmarks_vec == "error":
                pass
            else:
                testing_data.append(landmarks_vec)
                testing_label.append(emotions.index(emotion))

    return training_data, training_label, testing_data, testing_label

def create_model():
    accur_lin = []
    max_accur = 0
    for i in range(0,1):
        #Make sets by random sampling 80/20%
        print("Marking set %s" %i)
        X_train, y_train, X_test, y_test = make_sets()

        #Turn the training set into a numpy array for the classifier
        np_train = np.array(X_train)
        np_test = np.array(y_train)
        #train SVM
        print("Trainging SVM Classifier %s" %i)
        clf.fit(np_train, np_test)

        #Use score() function to get accuracy
        print("Getting accuracy score -- %s" %i)
        #npar_pred = np.array(X_test)
        pred_lin = clf.score(X_test, y_test)

        #Find Best Accuracy and save to file
        if pred_lin > max_accur:
            max_accur = pred_lin
            max_clf = clf

        print("Test Accuracy: ", pred_lin)
        accur_lin.append(pred_lin)  #Store accuracy in a list

    print("Mean Accuracy Value: %.3f" %np.mean(accur_lin))   #Get mean accuracy of the 10 runs

    predictions = max_clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    return max_accur, max_clf

if __name__ == '__main__':
    max_accur, max_clf = create_model()
    print('Best accuracy = ', max_accur*100, 'percent')
    print(max_clf)
    try:
        os.remove('models\model1.pkl')
    except OSError:
        pass
    output = open('models\model1.pkl', 'wb')
    pickle.dump(max_clf, output)
    output.close()
