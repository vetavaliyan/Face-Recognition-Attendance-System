import cv2
import face_recognition
import os
from datetime import datetime

path = 'ImagesAttendance'

myList = os.listdir(path)
images = [cv2.imread(os.path.join(path, cl)) for cl in myList]
classNames = [os.path.splitext(cl)[0] for cl in myList]
print(classNames)

attendance = set()

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encodeList.append(face_encodings[0])
    return encodeList


def markAttendance(name):
    if name not in attendance:
        with open('Attendance.csv', 'a') as f:
            if f.tell() == 0:  # Check if the file is empty
                f.write("Name,Date,Time\n")  # Write the headings
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{now.strftime("%Y-%m-%d")},{dtString}\n')
            attendance.add(name)


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    try:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for faceLoc, encodeFace in zip(facesCurFrame, encodesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = faceDis.argmin()

            name = 'Unknown'
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = [loc * 4 for loc in faceLoc]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    except KeyboardInterrupt:
        break

    except Exception as e:
        print('An error occurred:', e)

cap.release()
cv2.destroyAllWindows()



import cv2
import face_recognition

imgLebron = face_recognition.load_image_file('ImagesBasic/Lebron James.jpeg')
imgLebron = cv2.cvtColor(imgLebron, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Lebron James.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgLebron)[0]
encodeLebron = face_recognition.face_encodings(imgLebron)[0]
cv2.rectangle(imgLebron, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeLebron], encodeTest)
faceDis = face_recognition.face_distance([encodeLebron], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Lebron James', imgLebron)
cv2.imshow('Lebron James', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()


