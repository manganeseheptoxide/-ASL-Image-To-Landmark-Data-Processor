# THE THREADING MODULE MESSES UP THE LANDMARK COORDINATES AND IDK WHY
# THE ONLY USABLE DATA HERE ARE THE CENTERED ONES
# IT IS SLOW AND NOT INTENDED FOR DATA COLLECTION PURPOSES

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import threading
import queue
import time
from data_processing import *
from data_collection import *
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from image_feed import ImageFeed

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


def detect():
    global _landmarks_list, _landmark_connections, _status, _sample
    image_folder = f'raw_data\{_sample}-samples'
    cap = ImageFeed(image_folder = image_folder, loop = False)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
    
            ret, frame = cap.read()
            if not ret:
                _status = False
                break
            image, _landmarks_list, _landmark_connections = detect_upperbody(frame, hands)
            if _landmarks_list and _landmarks_list.landmark:
                mp_drawing.draw_landmarks(image, _landmarks_list, _landmark_connections)
                
                # Calculate the bounding box
                h, w, _ = image.shape
                buffer = 0.1
                x_min = min([landmark.x for landmark in _landmarks_list.landmark]) * w * (1 - buffer)
                y_min = min([landmark.y for landmark in _landmarks_list.landmark]) * h * (1 - buffer)
                x_max = max([landmark.x for landmark in _landmarks_list.landmark]) * w * (1 + buffer)
                y_max = max([landmark.y for landmark in _landmarks_list.landmark]) * h * (1 + buffer)
                
                # Draw the bounding box
                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                landmark_list = center_xyzlandmarks(_landmarks_list)
                _data_cache.put(landmark_list)
            
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                _status = False
                break

    cap.release()
    cv2.destroyAllWindows()


for letter in range(ord('A'), ord('Z') + 1):
  _landmarks_list, _landmark_connections, _centered_landmark_df = None, None, create_xyz_landmark_df()
  _status, _data_cache = True, queue.Queue()
  _sample = chr(letter)
  detect()

  print('Please Wait')
  time.sleep(1)

  df = df_entry_from_queue_NLL(df = _centered_landmark_df, data_cache = _data_cache)
  file_name = f"processed_data\{_sample}_data.csv"
  create_csv(df = df, file_name=file_name)



