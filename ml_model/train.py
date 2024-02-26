import cv2
import intel-numpy as np
import os

people = ['Rhythm']
dir_path = r"training_data"
features = []
labels = []

# Load Haar cascade classifier
haar_cascade_path = r"lol.xml"
haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

def create_train():
    for person in people:
        # person_dir = os.path.join(dir_path, person)
        label = people.index(person)

        for img_name in os.listdir(dir_path):
            print("DEBUG!!: ", img_name)
            img_path = os.path.join(dir_path, img_name)
            print(img_path)
            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x, y, w, h) in face_rect:
                faces_roi = gray[y:y+h, x:x+w]
                # Resize the face region to a fixed size (e.g., 100x100)
                faces_roi_resized = cv2.resize(faces_roi, (100, 100))
                features.append(faces_roi_resized)
                labels.append(label)

create_train()
print('Training DOne -----------------')

features = np.array(features)
labels = np.array(labels)

# Create LBPH face recognizer
face_recog = cv2.face.LBPHFaceRecognizer_create()
face_recog.train(features, labels)

face_recog.save('lol.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
