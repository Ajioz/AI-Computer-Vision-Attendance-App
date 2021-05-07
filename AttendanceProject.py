import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'
images = []
class_names = []

my_list = os.listdir(path)
# print(my_list)

for classifier in my_list:
    current_image = cv2.imread(f'{path}/{classifier}')
    images.append(current_image)
    class_names.append(os.path.splitext(classifier)[0])
print(class_names)

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


def mark_attendance(reg_name):
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if reg_name not in name_list:
            now = datetime.now()
            date_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{reg_name}, {date_string}')


#This serves as our encoded faces database
encode_list_known = find_encodings(images)
print("Encoding Complete")

#Turn on web cam
capture = cv2.VideoCapture(0)

while True:
    #read from the webcam
    success, img = capture.read()

    #resize the read image form webcam
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    #Convert to RGB for consistency
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    #Locate face as camera can capture multiple faces at once
    face_current_frame = face_recognition.face_locations(img_small)

    #Encode the face captured from webcam
    encode_current_frame = face_recognition.face_encodings(img_small, face_current_frame)

    #Loop through the captured face and encoded frames in other to compare with the database
    for encoded_face, face_location in zip(encode_current_frame, face_current_frame):
        matches = face_recognition.compare_faces(encode_list_known, encoded_face)
        face_distance = face_recognition.face_distance(encode_list_known, encoded_face)
        # print(face_distance)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = class_names[match_index].upper()
            # print(name)
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2, y2), (100, 55, 150), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (100, 55, 150), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)