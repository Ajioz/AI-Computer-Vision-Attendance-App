import cv2
import numpy as np
import face_recognition


img_check = face_recognition.load_image_file("ImagesBasic/Elon Musk.jpg")
img_check = cv2.cvtColor(img_check, cv2.COLOR_BGR2RGB)

img_test = face_recognition.load_image_file("ImagesBasic/Elon Test.jpg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(img_check)[0]
encode_img = face_recognition.face_encodings(img_check)[0]
cv2.rectangle(img_check, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloc_test = face_recognition.face_locations(img_test)[0]
encode_test = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(img_test, (faceloc_test[3], faceloc_test[0]), (faceloc_test[1], faceloc_test[2]), (255, 0, 0), 2)

result = face_recognition.compare_faces([encode_img], encode_test)
face_dis = face_recognition.face_distance([encode_img], encode_test)
print(result, face_dis)
cv2.putText(img_test, f'{result} {round(face_dis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Captured Image", img_check)
cv2.imshow("Test Image", img_test)
cv2.waitKey(0)