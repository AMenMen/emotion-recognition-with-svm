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
#emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]
emotions = ["anger", "disgust", "happy", "neutral", "surprise"]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  #Or set this to whatever you named the downloaded file
clf = SVC(C=0.001, kernel='linear', decision_function_shape='ovo', probability=True)   #Set the classifier as a support vector machines with polynomial kernel

#Define function to get file list, randomly shuffle it and split 80/20
def get_files(emotion):
    files = glob.glob("dataset_combine\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)]   #get first 80% of file list
    prediction = files[-int(len(files)*0.2):]   #get last 20% of file list
    return training, prediction

def get_landmarks(image):
    detections = detector(image, 1)
    #For all detected face instances individually
    for k,d in enumerate(detections):
        #Draw Facial Landmarks with the predictor class
        shape = predictor(image, d)
        xlist = []
        ylist = []
        for i in range(17, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        #Find both coordinates of centre of gravity
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        #Calculate distance centre <-> other points in both axes
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
        if xlist[11] == xlist[14]:
            anglenose = 0
        else:
            #point 29 is the tip of the nose, point 26 is the top of the nose brigde
            anglenose = int(math.atan((ylist[11]-ylist[14])/(xlist[11]-xlist[14]))*180/math.pi)

        #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x,y,w,z in zip(xcentral, ycentral, xlist, ylist):
            #Add the coordinates relative to the centre of gravity
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)

            #Get the euclidean distance between each point and the centre point (the vector length)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            #print(w-xmean)
            #Get the angle the vector describes relative to the image, corrected for the offset that the nosebrigde has when the face is not perfectly horizontal
            if w == xmean:
                anglerelative = 0 - anglenose
            else:
                anglerelative = (math.atan(float(z-ymean)/(w-xmean))*180/math.pi) - anglenose
                #print(anglerelative)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)
        #print('Length x: %d' % len(xcentral))

    if len(detections) < 1:
        #If no face is detected set the data to value "error" to catch detection errors
        landmarks_vectorised = "error"
    return landmarks_vectorised

def make_sets():
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        training, prediction = get_files(emotion)
        #Append data to traing and prediction list, and generate labels 0-7
        for item in training:
            #open image
            image = cv2.imread(item)
            #convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            #print("Hello 1")
            if landmarks_vectorised == "error":
                pass
            else:
                #append image array to trainging data list
                training_data.append(landmarks_vectorised)
                training_labels.append(emotions.index(emotion))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            landmarks_vectorised = get_landmarks(clahe_image)
            #print("Hello 2")
            if landmarks_vectorised == "error":
                pass
            else:
                prediction_data.append(landmarks_vectorised)
                prediction_labels.append(emotions.index(emotion))

    return training_data, training_labels, prediction_data, prediction_labels

def create_model():
    accur_lin = []
    max_accur = 0
    for i in range(0,1):
        #Make sets by random sampling 80/20%
        print("Marking set %s" %i)
        X_train, y_train, X_test, y_test = make_sets()

        #Turn the training set into a numpy array for the classifier
        npar_train = np.array(X_train)
        npar_trainlabs = np.array(y_train)
        #train SVM
        print("Trainging SVM Classifier %s" %i)
        clf.fit(npar_train, npar_trainlabs)

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
