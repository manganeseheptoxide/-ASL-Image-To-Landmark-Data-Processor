import mediapipe as mp
import cv2
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

NormLandmark = mp.tasks.components.containers.NormalizedLandmark
NormLandmarkList = landmark_pb2.NormalizedLandmarkList

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

useful_pose_landmarks_index = [0, 7, 8, 9, 10, 11, 12, 13, 14]

_hand_connections_1 = [pair for pair in mp_hands.HAND_CONNECTIONS]
_hand_connections_2 = [(pair[0] + 21, pair[1] + 21) for pair in mp_hands.HAND_CONNECTIONS]
_pose_connections_nohands = [(0, 1), (0, 2), (1, 2),
                             (0, 3), (0, 4), (3, 4),
                             (5, 6), (5, 7), (6, 8),
                             (1, 3), (2, 4)]
_pose_connections_1hand = [(pair[0] + 21, pair[1] + 21) for pair in _pose_connections_nohands]
_pose_connections_2hands = [(pair[0] + 42, pair[1] + 42) for pair in _pose_connections_nohands]

def get_connection_list(num_hands: int = 2, pose: bool = True):
    if num_hands == 0 and pose:
        out = _pose_connections_nohands
    elif num_hands == 1:
        out = _hand_connections_1 + _pose_connections_1hand if pose else _hand_connections_1
    elif num_hands == 2:
        out = _hand_connections_1 + _hand_connections_2 + _pose_connections_2hands + [(0, 50), (21, 49)] if pose else _hand_connections_1 + _hand_connections_2
    else:
        out = []
    return out

def connections(num_hands: int = 0, classification: list = [], pose: bool = False):
    global _hand_connections_1, _hand_connections_2, _pose_connections_nohands, _pose_connections_1hand, _pose_connections_2hands
    
    # Creates a list of tuples of landmark connections based on the detected data

    if num_hands == 0 and pose:
        out = _pose_connections_nohands
    elif num_hands == 1:
        out = _hand_connections_1 + _pose_connections_1hand if pose else _hand_connections_1
    elif num_hands == 2:
        out = _hand_connections_1 + _hand_connections_2 + _pose_connections_2hands if pose else _hand_connections_1 + _hand_connections_2
    else:
        out = []

    if num_hands != 0 and pose and classification:
        if num_hands == 1:
            out += [(0, 29)] if classification[0] == 'left' else [(0, 28)]
        elif num_hands == 2:
            out += [(0, 50), (21, 49)] # if classification[0] == 'left' else [(0, 49), (21, 50)]
        else:
            pass

    return out

def detect_upperbody(frame, hands, only_52: bool = False, hand_priority: str = 'right'):
    # Check if the frame is empty
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty")

    # modified detection using hands and pose detection
    # returns the image, landmarks, and the appropriate connections

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    hands_results = hands.process(image)
    # pose_results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    container = []
    hand_used = None

    num_hands_detected = 0 if bool(hands_results.multi_hand_landmarks) == False else len(hands_results.multi_hand_landmarks)
    hand_classifications = [side.classification[0].label.lower() for side in hands_results.multi_handedness] if bool(hands_results.multi_handedness) else []
    # pose_detected = bool(pose_results.pose_landmarks)
    hand_used = ''
    if hands_results.multi_hand_landmarks:
        if num_hands_detected == 1:
            container = list(hands_results.multi_hand_landmarks[0].landmark)
            hand_used = hand_classifications[0]
        
        else:
            if hand_priority == 'left':
                for i, landmarks in enumerate(hands_results.multi_hand_landmarks):
                    if hand_classifications[i] == 'left':
                        container = list(landmarks.landmark)
                        hand_used = 'left'
                        break
                if not container:
                    container = list(hands_results.multi_hand_landmarks[0].landmark)
                    hand_used = hand_classifications[0]
            else:
                for i, landmarks in enumerate(hands_results.multi_hand_landmarks):
                    if hand_classifications[i] == 'right':
                        container = list(landmarks.landmark)
                        hand_used = 'right'
                        break
                if not container:
                    container = list(hands_results.multi_hand_landmarks[0].landmark)
                    hand_used = hand_classifications[0]

    if only_52 and container:
        useful_landmarks = NormLandmarkList(landmark = container) if len(container) == 52 else NormLandmarkList()
    else:
        useful_landmarks = NormLandmarkList(landmark = container) if container else NormLandmarkList()

    connection_data = connections(num_hands = 1, classification = hand_classifications, pose = False) if useful_landmarks.landmark else []
    
    # image, NormalizedLandmarkList, list, and hand used
    return image, useful_landmarks, connection_data, hand_used

def landmarklist_to_xyzcoord(landmark_list, all_52 : bool = False, centered: bool = False, normalize: bool = False):
    # landmark_list must be a NormalizedLandmarkList Object
    if centered:
        landmark_list = center_xyzlandmarks(landmark_list, normalized=normalize)
    if landmark_list:
        if all_52:
            landmark_coordiantes = [[landmark.x, landmark.y, landmark.z] for landmark in landmark_list.landmark] if len(landmark_list.landmark) == 52 else []
        else:
            landmark_coordiantes = [[landmark.x, landmark.y, landmark.z] for landmark in landmark_list.landmark]
    else:
        return []
    # converts a NormalizedLandmarkList with n NormalizedLandmarks into a list in the format of [[x1, y1, z1],...,[xn, yn, zn]] 
    return landmark_coordiantes
def center_xyzlandmarks(landmarks_list, normalized: bool = True):
    if landmarks_list and landmarks_list.landmark:
        x_center = landmarks_list.landmark[9].x
        y_center = landmarks_list.landmark[9].y
        z_center = landmarks_list.landmark[9].z
        x_norm = (landmarks_list.landmark[0].x - x_center)
        y_norm = (landmarks_list.landmark[0].y - y_center)
        z_norm = (landmarks_list.landmark[0].z - z_center)
        norm0 = (x_norm**2 + y_norm**2 + z_norm**2)**0.5
        for landmark in landmarks_list.landmark:
            landmark.x = (landmark.x - x_center)/norm0 if normalized else landmark.x - x_center
            landmark.y = (landmark.y - y_center)/norm0 if normalized else landmark.y - y_center
            landmark.z = (landmark.z - z_center)/norm0 if normalized else landmark.z - z_center
        
    return landmarks_list




