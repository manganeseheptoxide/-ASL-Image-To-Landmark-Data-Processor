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


# _landmarks_list, _landmark_connections, _centered_landmark_df = None, None, create_xyz_landmark_df()
# _status, _data_cache = False, queue.Queue()
# _sample = 'A'
# image_folder = f'raw_data\ASL_Data\{_sample}-samples'
# cap = ImageFeed(image_folder = image_folder, loop = False)
# cap = cv2.VideoCapture(0)

def plot_landmarks_modified(empty = None, elevation: int = 10, azimuth: int = 10):

  # This is a modified version of the built-in function mediapipe.solutions.drawing_utils.plot_landmarks
  # The original function can be found on *path*\python\python38\lib\site-packages\mediapipe\python\solutions\drawing_utils.py
  # The modifications in this code was done so that it will be able to dispaly a live feed of the landmarks

  global _landmarks_list, _landmark_connections, _data_cache
  landmark_list = center_xyzlandmarks(_landmarks_list)
  _data_cache.put(landmark_list)
  connections0 = _landmark_connections
  if not landmark_list:
    return

  plt.cla()
  ax.view_init(elev=elevation, azim=azimuth)
  plotted_landmarks = {}
  for idx, landmark in enumerate(landmark_list.landmark):
 
    ax.scatter3D(
        xs=[-landmark.z],
        ys=[landmark.x],
        zs=[-landmark.y],
        color='r')
    plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
  if connections0:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections0:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
        landmark_pair = [
            plotted_landmarks[start_idx], plotted_landmarks[end_idx]
        ]
        ax.plot3D(
            xs=[landmark_pair[0][0], landmark_pair[1][0]],
            ys=[landmark_pair[0][1], landmark_pair[1][1]],
            zs=[landmark_pair[0][2], landmark_pair[1][2]],
            color='g')
  lim = 0.5
  ax.set_xlim(-lim, lim)   # Set x-axis limits
  ax.set_ylim(-lim, lim)   # Set y-axis limits
  ax.set_zlim(-lim, lim)   # Set z-axis limits

def detect():
    global _landmarks_list, _landmark_connections, _status, _sample
    image_folder = f'raw_data\ASL_Data\{_sample}-samples'
    cap = ImageFeed(image_folder = image_folder, loop = False)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            # global _landmarks_list, _landmark_connections, _data_cache
            # landmark_list = center_xyzlandmarks(_landmarks_list)
            # _data_cache.put(landmark_list)
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

def record():
    global _status, _data_cache
    while _status:
        if not _data_cache.empty():
            print(_data_cache.get())
        time.sleep(0.5)

# _status = True

# detect()
# threading.Thread(target = detect).start()
for letter in range(ord('A'), ord('Z') + 1):
  _landmarks_list, _landmark_connections, _centered_landmark_df = None, None, create_xyz_landmark_df()
  _status, _data_cache = True, queue.Queue()
  _sample = chr(letter)
  detect()
  # print(chr(letter), end="")  # Output: A B C ... Z

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# animate = FuncAnimation(plt.gcf(), plot_landmarks_modified, interval=50)
# plt.show()

# while _status:
#    continue

  print('Please Wait')
  time.sleep(1)

  df = df_entry_from_queue_NLL(df = _centered_landmark_df, data_cache = _data_cache)
  file_name = f"processed_data\{_sample}_data.csv"
  create_csv(df = df, file_name=file_name)



