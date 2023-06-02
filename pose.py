import cv2
import mediapipe as mp
import numpy as np
from numpy import dot
from numpy.linalg import norm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray

#Opening vector articulations
pose1_1=np.load('16_22.npy')
pose1_2=np.load('16_20.npy')
pose1_3=np.load('16_18.npy')
pose1_4=np.load('16_14.npy')
pose1_5=np.load('14_12.npy')
pose1_6=np.load('15_21.npy')
pose1_7=np.load('15_19.npy')
pose1_8=np.load('15_17.npy')
pose1_9=np.load('15_13.npy')
pose1_10=np.load('13_11.npy')
pose1_11=np.load('12_24.npy')
pose1_12=np.load('24_26.npy')
pose1_13=np.load('26_28.npy')
pose1_14=np.load('28_30.npy')
pose1_15=np.load('28_32.npy')
pose1_16=np.load('11_23.npy')
pose1_17=np.load('23_25.npy')
pose1_18=np.load('25_27.npy')
pose1_19=np.load('27_29.npy')
pose1_20=np.load('27_31.npy')

pose1_com=np.load('pose1_complete.npy')

def cosine_similarity(matrix1, matrix2):
    # Compute the dot product of the two matrices
    dot_product = np.dot(matrix1, matrix2.T)
    
    # Calculate the magnitudes of each matrix
    magnitude1 = np.sqrt(np.sum(np.square(matrix1), axis=1))
    magnitude2 = np.sqrt(np.sum(np.square(matrix2), axis=1))
    
    # Compute the cosine similarity
    similarity = dot_product / np.outer(magnitude1, magnitude2)
    
    return similarity

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    height, width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    right_in = int ()
    
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  
  
  while cap.isOpened():
    success, image = cap.read()
    height, width, _ = image.shape

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
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
        
  #Determinnar los puntos de las articulaciones
        
      #brazo derecho
      veinte_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x)
      veinte_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y)
        
      dieciocho_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x)
      dieciocho_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y)

      veintidos_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x)
      veintidos_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y)  

      dieciseis_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)
      dieciseis_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y)

      catorce_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
      catorce_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y)

      doce_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
      doce_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

      #Torso del lado derech
      veinticuatro_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
      veinticuatro_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
          
      veintiseis_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
      veintiseis_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)

      #pie lado derecho
      veintiocho_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x)
      veintiocho_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

      treinta_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x)
      treinta_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y)

      treintaydos_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
      treintaydos_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)

      #brazo izquierdo
      diecinueve_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x)
      diecinueve_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y)

      diecisiete_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x)
      diecisiete_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y)

      veintiuno_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x)
      veintiuno_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y)

      quince_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
      quince_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)

      trece_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x)
      trece_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y)

      once_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
      once_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
          
      #Torso del lado izquierdo
      veintitres_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x)
      veintitres_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y)

      veinticinco_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
      veinticinco_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)

      #pie lado izquierdo
      veintisiete_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x)
      veintisiete_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)

      veintinueve_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x)
      veintinueve_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y)

      treintayuno_x=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
      treintayuno_y=(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
        
      print('                                                        ')

        #Determinar vectores dependiendo de la pose 
      dieciseis_veintidos= np.array([dieciseis_x, dieciseis_y]) - np.array([veintidos_x, veintidos_y])
      result = dot(dieciseis_veintidos, pose1_1)/(norm(dieciseis_veintidos)*norm(pose1_1))*100
      if 95<result<100:
        image=cv2.line(image, (int(doce_x*width), int(doce_y*height)), (int(catorce_x*width), int(catorce_y*height)), (0,255,0), 5)
      
      dieciseis_veinte=np.array([dieciseis_x, dieciseis_y]) - np.array([veinte_x, veinte_y])
      result2=dot(dieciseis_veinte, pose1_2)/(norm(dieciseis_veinte)*norm(pose1_2))*100
      if 95<result2<100:
        image=cv2.line(image, (int(dieciseis_x*width), int(dieciseis_y*height)), (int(veinte_x*width), int(veinte_y*height)), (0,255,0), 5)
      # print(dieciseis_veinte)
      
      dieciseis_dieciocho=np.array([dieciseis_x, dieciseis_y]) - np.array([dieciocho_x, dieciocho_y]) 
      result3=dot(dieciseis_dieciocho, pose1_3)/(norm(dieciseis_dieciocho)*norm(pose1_3))*100
      if 95<result3<100:
        image=cv2.line(image, (int(dieciseis_x*width), int(dieciseis_y*height)), (int(dieciocho_x*width), int(dieciocho_y*height)), (0,255,0), 5)
      # print(dieciseis_dieciocho)
       
      dieciseis_catorce=np.array([dieciseis_x, dieciseis_y]) - np.array([catorce_x, catorce_y])
      result4=dot(dieciseis_catorce, pose1_4)/(norm(dieciseis_catorce)*norm(pose1_4))*100
      if 95<result4<100:
        image=cv2.line(image, (int(dieciseis_x*width), int(dieciseis_y*height)), (int(catorce_x*width), int(catorce_y*height)), (0,255,0), 5)
      # print(dieciseis_catorce)
      
      catorce_doce=np.array([catorce_x, catorce_y]) - np.array([doce_x, doce_y])
      result5=dot(catorce_doce, pose1_5)/(norm(catorce_doce)*norm(pose1_5))*100
      if 95<result5<100:
        image=cv2.line(image, (int(catorce_x*width), int(catorce_y*height)), (int(doce_x*width), int(doce_y*height)), (0,255,0), 5)
      # print(catorce_doce)
      
      quince_veintiuno=np.array([quince_x, quince_y]) - np.array([veintiuno_x, veintiuno_y])
      result6=dot(quince_veintiuno, pose1_6)/(norm(quince_veintiuno)*norm(pose1_6))*100
      if 95<result6<100:
        image=cv2.line(image, (int(quince_x*width), int(quince_y*height)), (int(veintiuno_x*width), int(veintiuno_y*height)), (0,255,0), 5)
      
      quince_diecinueve=np.array([quince_x, quince_y]) - np.array([diecinueve_x, diecinueve_y])
      result7=dot(quince_diecinueve, pose1_7)/(norm(quince_diecinueve)*norm(pose1_7))*100
      if 95<result7<100:
        image=cv2.line(image, (int(quince_x*width), int(quince_y*height)), (int(diecinueve_x*width), int(diecinueve_y*height)), (0,255,0), 5)
      # print(quince_diecinueve)
      
      quince_diecisiete=np.array([quince_x, quince_y]) - np.array([diecisiete_x, diecisiete_y])
      result8=dot(quince_diecisiete, pose1_8)/(norm(quince_diecisiete)*norm(pose1_8))*100
      if 95<result8<100:
        image=cv2.line(image, (int(quince_x*width), int(quince_y*height)), (int(diecisiete_x*width), int(diecisiete_y*height)), (0,255,0), 5)
      # print(quince_diecisiete)
      
      quince_trece=np.array([quince_x, quince_y]) - np.array([trece_x, trece_y])
      result9=dot(quince_trece, pose1_9)/(norm(quince_trece)*norm(pose1_9))*100
      # print(result9)
      if 95<result9<100:
        image=cv2.line(image, (int(quince_x*width), int(quince_y*height)), (int(trece_x*width), int(trece_y*height)), (0,255,0), 5)
      # print(quince_trece)
      
      trece_once=np.array([trece_x, trece_y]) - np.array([once_x, once_y])
      result10=dot(trece_once, pose1_10)/(norm(trece_once)*norm(pose1_10))*100
      # print(result10)
      if 95<result10<100:
        image=cv2.line(image, (int(trece_x*width), int(trece_y*height)), (int(once_x*width), int(once_y*height)), (0,255,0), 5)
        
      #VECTORES DEL TORSO 
      #derecho      
      doce_veinticuatro=np.array([doce_x, doce_y]) - np.array([veinticuatro_x, veinticuatro_y])
      result11=dot(doce_veinticuatro, pose1_11)/(norm(doce_veinticuatro)*norm(pose1_11))*100
      # print(result11)
      if 95<result11<100:
        image=cv2.line(image, (int(doce_x*width), int(doce_y*height)), (int(veinticuatro_x*width), int(veinticuatro_y*height)), (0,255,0), 5)
      # print(doce_veinticuatro)
      
      veinticuatro_veintiseis=np.array([veinticuatro_x, veinticuatro_y]) - np.array([veintiseis_x, veintiseis_y])
      result12=dot(veinticuatro_veintiseis, pose1_12)/(norm(veinticuatro_veintiseis)*norm(pose1_12))*100
      # print(result12)
      if 95<result12<100:
        image=cv2.line(image, (int(veinticuatro_x*width), int(veinticuatro_y*height)), (int(veintiseis_x*width), int(veintiseis_y*height)), (0,255,0), 5)
      # print(veinticuatro_veintiseis)
      
      veintiseis_veintiocho=np.array([veintiseis_x, veintiseis_y]) - np.array([veintiocho_x, veintiocho_y])
      result13=dot(veintiseis_veintiocho, pose1_13)/(norm(veintiseis_veintiocho)*norm(pose1_13))*100
      # print(result13)
      if 95<result13<100:
        image=cv2.line(image, (int(veintiseis_x*width), int(veintiseis_y*height)), (int(veintiocho_x*width), int(veintiocho_y*height)), (0,255,0), 5)
      # print(veintiseis_veintiocho)
      
      veintiocho_treinta=np.array([veintiocho_x, veintiocho_y]) - np.array([treinta_x, treinta_y])
      result14=dot(veintiocho_treinta, pose1_14)/(norm(veintiocho_treinta)*norm(pose1_14))*100
      # print(result14)
      if 95<result14<100:
        image=cv2.line(image, (int(veintiocho_x*width), int(veintiocho_y*height)), (int(treinta_x*width), int(treinta_y*height)), (0,255,0), 5)
      # print(veintiocho_treinta)
      
      veintiocho_treintaydos=np.array([veintiocho_x, veintiocho_y]) - np.array([treintaydos_x, treintaydos_y])
      result_15=dot(veintiocho_treintaydos, pose1_15)/(norm(veintiocho_treintaydos)*norm(pose1_15))*100
      # print(result_15)
      if 95<result_15<100:
        image=cv2.line(image, (int(veintiocho_x*width), int(veintiocho_y*height)), (int(treintaydos_x*width), int(treintaydos_y*height)), (0,255,0), 5)
      # print(veintiocho_treintaydos)
      
      #izquierdo
      once_veintitres=np.array([once_x, once_y]) - np.array([veintitres_x, veintitres_y])
      result_16=dot(once_veintitres, pose1_16)/(norm(once_veintitres)*norm(pose1_16))*100
      # print(result_16)
      if 95<result_16<100:
        image=cv2.line(image, (int(once_x*width), int(once_y*height)), (int(veintitres_x*width), int(veintitres_y*height)), (0,255,0), 5)
      # print(once_veintitres)
      
      veintitres_veinticinco=np.array([veintitres_x, veintitres_y]) - np.array([veinticinco_x, veinticinco_y])
      result_17=dot(veintitres_veinticinco, pose1_17)/(norm(veintitres_veinticinco)*norm(pose1_17))*100
      # print(result_17)
      if 95<result_17<100:
        image=cv2.line(image, (int(veintitres_x*width), int(veintitres_y*height)), (int(veinticinco_x*width), int(veinticinco_y*height)), (0,255,0), 5)
      # print(veintitres_veinticinco)
      
      veinticinco_veintisiete=np.array([veinticinco_x, veinticinco_y]) - np.array([veintisiete_x, veintisiete_y])
      result_18=dot(veinticinco_veintisiete, pose1_18)/(norm(veinticinco_veintisiete)*norm(pose1_18))*100
      if 95<result_18<100:
        image=cv2.line(image, (int(veinticinco_x*width), int(veinticinco_y*height)), (int(veintisiete_x*width), int(veintisiete_y*height)), (0,255,0), 5)
      # print(veinticinco_veintisiete)
      
      veintisiete_veintinueve=np.array([veintisiete_x, veintisiete_y]) - np.array([veintinueve_x, veintinueve_y])
      result_19=dot(veintisiete_veintinueve, pose1_19)/(norm(veintisiete_veintinueve)*norm(pose1_19))*100
      # print(result_19)
      if 95<result_19<100:
        image=cv2.line(image, (int(veintisiete_x*width), int(veintisiete_y*height)), (int(veintinueve_x*width), int(veintinueve_y*height)), (0,255,0), 5)
      # print(veintisiete_veintinueve)
      
      veintisiete_treintayuno=np.array([veintisiete_x, veintisiete_y]) - np.array([treintayuno_x, treintayuno_y])
      result_20=dot(veintisiete_treintayuno, pose1_20)/(norm(veintisiete_treintayuno)*norm(pose1_20))*100
      # print(result_20)
      if 95<result_20<100:
        image=cv2.line(image, (int(veintisiete_x*width), int(veintisiete_y*height)), (int(treintayuno_x*width), int(treintayuno_y*height)), (0,255,0), 5)
      # print(veintisiete_treintayuno)
    
        
      matrix=np.array([dieciseis_veintidos,
                    dieciseis_veinte, 
                    dieciseis_dieciocho,
                    dieciseis_catorce,
                    catorce_doce,
                    quince_veintiuno,
                    quince_diecinueve,
                    quince_diecisiete,
                    quince_trece,
                    trece_once,
                    doce_veinticuatro,
                    veinticuatro_veintiseis,
                    veintiseis_veintiocho,
                    veintiocho_treinta,
                    veintiocho_treintaydos,
                    once_veintitres,
                    veintitres_veinticinco,
                    veinticinco_veintisiete,
                    veintisiete_veintinueve,
                    veintisiete_treintayuno])
      # print(matrix)
      
    similarity_matrix = cosine_similarity(matrix, pose1_com)
    similarity=np.mean(similarity_matrix)
    
    
    scor_pose1=(result+result2+result3+result4+result5+result6+result7+result8+result9+result10+result11+result12+result13+result14+result_15+result_16+result_17+result_18+result_19+result_20)/20
    if 95<scor_pose1<100:
      print('score pose with vector',scor_pose1)
      print ('score pose with matrix', similarity)
      print('similarity matrix', similarity_matrix)

    
      # try: 
      #   resultado= dot(dieciseis_veintidos)
      #   print(resultado)
      # except Exception as e:
      #   print(e)
      #   print('no se pudo')
      #   continue
      # print(resultado)
        
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      
      break
    
cap.release()
cv2.destroyAllWindows()