import mediapipe as mp
import numpy as np
import cv2
import os
from os import listdir
from pathlib import Path
from PIL import Image



# Calling the pose solution from MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BG_COLOR = (192, 192, 192) # gray

#POSE GENERATOR 
def pose_detector(image, image2, pose):
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    if results.pose_landmarks:
      mp_drawing.draw_landmarks(
            image2,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      
    return results, image2
  
LISTA_IMPORTANTES = [[16, 22], [16, 20], [16, 18], [16, 14], [14, 12], [15, 21], [15, 19], [15, 17], [15, 13], [13, 11], [12, 24], [24, 26], [26, 28], [28, 30], [28, 32], [11, 23], [23, 25], [25, 27], [27, 29], [27, 31]]

def obtener_vector(lista_importantes, results):
        lista_vectores = []
        for i in lista_importantes:
          x1 = results.pose_landmarks.landmark[i[0]].x
          y1 = results.pose_landmarks.landmark[i[0]].y
          x2 = results.pose_landmarks.landmark[i[1]].x
          y2 = results.pose_landmarks.landmark[i[1]].y
          vector = np.array([x1, y1]) - np.array([x2, y2])
          lista_vectores.append(vector)
        lista_vectores=np.array(lista_vectores).reshape(40)
        return lista_vectores
    
black = cv2.imread("fondo.jpg")
image = cv2.imread("paso8.jpg")
# get the path or directory
    
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
  
  
    # image = cv2.imread(image)
  height, width, _ = image.shape
  results, stick_man = pose_detector(image, black, pose)
  pose= obtener_vector(LISTA_IMPORTANTES, results)
    
  
np.save('pose9.npy', pose)
cv2.imwrite('stickman9.jpg', stick_man)

