import pickle
import threading
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound 
from time import sleep

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

def speak(predicted_character):
    if(predicted_character == '0'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\0.mp3")
    elif(predicted_character == '1'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\1.mp3")
    elif(predicted_character == '2'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\2.mp3")
    elif(predicted_character == '3'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\3.mp3")
    elif(predicted_character == '4'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\4.mp3")
    elif(predicted_character == '5'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\5.mp3")
    elif(predicted_character == '6'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\6.mp3")
    elif(predicted_character == '7'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\7.mp3")
    elif(predicted_character == '8'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\8.mp3")
    elif(predicted_character == '9'):
        playsound("C:\\Users\\sukes\\Downloads\\hand gesture\\9.mp3")
            

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: '0', 1: '1', 2: '2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0),4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        t1 = threading.Thread(target = speak, args = (predicted_character))
        t1.start()
        t1.join()
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
