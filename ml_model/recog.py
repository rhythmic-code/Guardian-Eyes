# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Face Detection
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# # Initialize OpenCV Face Recognition
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('lol.yml')  # Load trained face recognition model

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     # Read frame from webcam
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame")
#         break
    
#     # Convert the image to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Detect faces using MediaPipe
#     results = face_detection.process(rgb_frame)
    
#     # Draw rectangles around detected faces
#     if results.detections:
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             ih, iw, _ = frame.shape
#             bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
#                    int(bboxC.width * iw), int(bboxC.height * ih)
#             # cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            
#             # Convert face region to grayscale for recognition
#             gray_face = cv2.cvtColor(frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], cv2.COLOR_BGR2GRAY)
            
#             # Recognize face using OpenCV face recognizer
#             label, confidence = face_recognizer.predict(gray_face)
#             if confidence < 100:
#                 cv2.putText(frame, f'Person {label}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                 cv2.rectangle(frame, bbox, (0, 255, 0), 2)
#             else:
#                 cv2.putText(frame, 'Unknown', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#                 cv2.rectangle(frame, bbox, (0, 0, 255), 2)
    
#     # Display the resulting frame
#     cv2.imshow('Face Detection and Recognition', frame)
    
#     # Exit when 'e' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('e'):
#         break

# # Release the webcam and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
from openvino.inference_engine import IECore
import subprocess

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize OpenVINO Inference Engine
ie = IECore()
model_xml = 'D:/LVR/venv/lol.xml'  # Replace with path to the XML file of your face recognition model
# model_bin = 'lol.bin'  # Replace with path to the BIN file of your face recognition model
net = ie.read_network(model=model_xml)
exec_net = ie.load_network(network=net, device_name='CPU')

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MediaPipe
    results = face_detection.process(rgb_frame)
    
    # Draw rectangles around detected faces
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Convert face region to grayscale for recognition
            gray_face = cv2.cvtColor(frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], cv2.COLOR_BGR2GRAY)
            
            # Perform inference for face recognition using OpenVINO
            # Modify this part based on the specifics of your face recognition model
            # You'll need to extract features from the face region and compare them with known identities
            output = exec_net.infer(inputs={'input': gray_face})
            # Process the output and determine the label and confidence
            
            label = output['label']  # Determine the label based on the output
            confidence = output['confidence']  # Determine the confidence based on the output
            
            if confidence < 100:
                cv2.putText(frame, f'Person {label}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                # Send ping if face detected
                subprocess.run(["ping", "-c", "1", "your_host_here"])  # Replace "your_host_here" with the host you want to ping
            else:
                cv2.putText(frame, 'Unknown', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, bbox, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Detection and Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
