#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
import mediapipe as mp 
import socket
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_array = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10","11","12","13","14","15","16"]

LISTA_IMPORTANTES = [[16, 22], [16, 20], [16, 18], [16, 14], [14, 12], [15, 21], [15, 19], [15, 17], [15, 13], [13, 11], [12, 24], [24, 26], [26, 28], [28, 30], [28, 32], [11, 23], [23, 25], [25, 27], [27, 29], [27, 31]]

# For static images:
BG_COLOR = (192, 192, 192) # gray

data1 = "0"
data2 = "0"
ind_img = "1"

def socket_connection():
  global data1, data2, ind_img
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect(('10.22.130.219', 5001))
  prev_message = "NO"
  while(True):
    # #MANDANDO PRIMERA ITERACIÓN EN 0
    data = str(ind_img) + " " + str(data1) + " " + str(data2)
    s.sendall(data.encode())
    message = s.recv(1024).decode()

    #OBTENER LA POSE DE LA ITERACION ALEATORIA
    if (message == "YES" and prev_message == "YES"):
      i = random.randint(0, len(pose_array)-1)
      ind_img = pose_array[i]
      print("CHANGING ITERATION")
      
    prev_message = message
    print(message)
    print(prev_message)
    


def media_pipe_video():
  global data1, data2, ind_img
  # Initialize the camera
  cap = cv2.VideoCapture(0)  # 0 represents the default camera, change it if needed
  dict = {"pose1_com":np.load('pose1.npy'), "pose2_com":np.load('pose2.npy'), "pose3_com":np.load('pose3.npy'), "pose4_com":np.load('pose4.npy'), "pose5_com":np.load('pose5.npy'), "pose6_com":np.load('pose6.npy'), "pose7_com":np.load('pose7.npy'), "pose8_com":np.load('pose8.npy'), "pose9_com":np.load('pose9.npy'), "pose10_com":np.load('pose10.npy'), "pose11_com":np.load('pose11.npy'), "pose12_com":np.load('pose12.npy'), "pose13_com":np.load('pose13.npy'), "pose14_com":np.load('pose14.npy'), "pose15_com":np.load('pose15.npy'), "pose16_com":np.load('pose16.npy')}

  def cosine_similarity(matrix1, matrix2):
      # Compute the dot product of the two matrices
    similarity=dot(matrix1, matrix2)/(norm(matrix1)*norm(matrix2))*100
    int(similarity)
      
    return similarity

  def pose_detector(image, pose):
      
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      
      if results.pose_landmarks:
        mp_drawing.draw_landmarks(
              image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
      return results, image

  def iluminate (image, lista_importantes, results):
          width, height, _ = image.shape
          for i in lista_importantes:
            image=cv2.line(image, (int(results.pose_landmarks.landmark[i[0]].x*width), int(results.pose_landmarks.landmark[i[0]].y*height)), (int(results.pose_landmarks.landmark[i[1]].x*width), int(results.pose_landmarks.landmark[i[1]].y*height)), (0,255,0), 5)
          return image

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



  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    
    
    while cap.isOpened():
      success, image = cap.read()
      height, width, _ = image.shape
      
      roi_width=int(width/2)
      
      #DIVIDIR LA IMAGEN EN DOS PARTES
      left_roi = image[:, :roi_width, :]
      right_roi = image[:, roi_width:, :]
      
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
      
      #SACAR LA POSE DEL LADO IZQUIERDO Y DEL DERECHO
    
      izq, left_roi=pose_detector(left_roi, pose)
      der, right_roi=pose_detector(right_roi, pose)
      
      #Comparar la pose del lado izquierdo y derecho, independientemente de cual se muestre
      if izq.pose_landmarks:
        pose_izq=obtener_vector(LISTA_IMPORTANTES, izq)
        derf_pose="pose"+ind_img+"_com"
        similarity_pose_izq=cosine_similarity(pose_izq, dict[derf_pose])
        data1=str(similarity_pose_izq)
      else:
        data1 = 0
      
      
      if der.pose_landmarks:
        pose_der=obtener_vector(LISTA_IMPORTANTES, der)
        of_pose="pose"+ind_img+"_com"
        similarity_pose_der=cosine_similarity(pose_der, dict[of_pose])
        data2=str(similarity_pose_der)
      else:
        data2 = 0
    

      cv2.imshow('player 1', left_roi)
      cv2.imshow('player 2', right_roi)
      

      if cv2.waitKey(5) & 0xFF == 27:
        
        break
      
      
  #MAQUINA DE ESTADOS: 
  #ITERE SOLO CUANDO LE MANDAS L
      
  cap.release()
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  loop = asyncio.new_event_loop()
  executor = ThreadPoolExecutor(10)
  video_capture = loop.run_in_executor(executor, media_pipe_video)
  socket_conn = loop.run_in_executor(executor, socket_connection)
