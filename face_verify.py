# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 00:54:17 2023

@author: amapande
"""


import cv2
import numpy as np
import face_recognition
import os
from skimage import exposure
import argparse


def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

def faceRecognition(input_folder_path,
                    source='0',
                    input_file_path='0', 
                    output_folder_path="./"):
    
    path = input_folder_path
    images = []
    classNames = []
    myList = os.listdir(path)
    # print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    if source=='0':
        cap = cv2.VideoCapture(0)
        
        while True:
            success, img = cap.read()
            p2, p98 = np.percentile(img, (3, 97))
            img1 = exposure.rescale_intensity(img, in_range=(p2, p98))
            #img = captureScreen()
            imgS = cv2.resize(img1,(0,0),None,0.25,0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
            
            for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                print(faceDis)
                matchIndex = np.argmin(faceDis)
            
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
                cv2.imshow('webcam',img)
            if cv2.waitKey(10) == ord('b'):
                break
        cap.release()
        cv2.destroyAllWindows()
    elif source=='1':
        img = cv2.imread(input_file_path)
        p2, p98 = np.percentile(img, (3, 97))
        img1 = exposure.rescale_intensity(img, in_range=(p2, p98))
        #img = captureScreen()
        imgS = cv2.resize(img1,(0,0),None,1,1)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            print(faceDis)
            matchIndex = np.argmin(faceDis)
            print(matches)
            print(matchIndex)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1 = faceLoc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            img_name= input_file_path.split("/")[-1]
            print(output_folder_path+"/"+img_name)
            cv2.imwrite(output_folder_path+"/"+img_name,img)
            print("image saved in the directory: "+ output_folder_path)         
            
def main(input_folder_path='./ImageFace',
         source='0',
         input_file_path='0', 
         output_folder_path="./face_detected"):
    


    faceRecognition(input_folder_path,
                    source,
                    input_file_path, 
                    output_folder_path)
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder_path", help="Image path of the person with visible face")
    parser.add_argument("source", help="Name of the person")
    parser.add_argument("-input_file_path", help="Image path of the person with visible face")
    parser.add_argument("-output_folder_path", help="Name of the person")
    args = parser.parse_args()
    input_file_path = ""
    output_folder_path="./face_detected"
    if not args.source=='0':
        input_file_path = args.input_file_path 
        if not args.output_folder_path is None:
            output_folder_path = args.output_folder_path

    main(args.input_folder_path,args.source,input_file_path,output_folder_path)

