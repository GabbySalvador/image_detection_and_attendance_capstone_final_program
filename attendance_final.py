import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'People_Folder_copy' #the path variable should hold the actual path of the folder where the images are inputted
list_people_images= []
image_names = []
myList = os.listdir(path)
print(myList)

#please refer to pseudocode documentation for more detailed steps on how the code was generated

for per in myList:       #reads all the images in the "People_Folder_copy"
    curImg = cv2.imread(f'{path}/{per}')
    list_people_images.append(curImg)
    image_names.append(os.path.splitext(per)[0])
print(image_names)

def determine_encodings(images):  #encodes these images into lists
    encodings_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings_list.append(encode)
    return encodings_list 

def markAttendance(name):  #this code was used to input the name of the image presented in the webcam in the csv file
    with open('attendance_people_final.csv','r+') as f:
        Data = f.readlines()
        list_of_names=[]
        for statement in Data:
            entry=statement.split(',')
            list_of_names.append(entry[0])

        if name not in list_of_names: 
            current_time=datetime.now()
            date_time_string=current_time.strftime('%H:%M:%S')
            month_year_string=current_time.strftime('%Y-%m-%d')
            f.writelines(f'\n{name},{date_time_string},{month_year_string}')



current_encoding_list=determine_encodings(list_people_images)
print("ENCODING COMPLETE")   

web_per = cv2.VideoCapture(0)
 
while True:
    success, img = web_per.read()           #this code resizes the images presented on the webcam before it gets encoded and compared to list of images in people folder
    resized_img = cv2.resize(img,(0,0),None,0.25,0.25)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
 
    faces_shown_web = face_recognition.face_locations(resized_img)
    encodes_shown_web = face_recognition.face_encodings(resized_img,faces_shown_web)

    for encodeFace,faceLoc in zip(encodes_shown_web ,faces_shown_web ):  #compares the face presented on webcam on the images on "People_Folder_copy"
        matches = face_recognition.compare_faces(current_encoding_list,encodeFace)
        faceDis = face_recognition.face_distance(current_encoding_list,encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex]< 0.50:       #this code returns the name of the person presented on webcam if his face is on the people's folder, and will return "Unknown" if it is not
            name =image_names[matchIndex].upper()
            markAttendance(name)
        else: name = 'Unknown'

        y1,x2,y2,x1 = faceLoc               #this code was used to generate the rectangular box which appears when face is presented on webcam
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(255, 153, 255),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(255, 153, 255),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

           
 
       
    
