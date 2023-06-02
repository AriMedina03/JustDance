import mediapipe as mp
import numpy as np
import cv2

# Calling the pose solution from MediaPipe
mp_pose = mp.solutions.pose

# Opening the image source to be used
image = cv2.imread("paso1.jpg")

# Calling the pose detection model
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # Detecting the pose with the image
    poseResult = pose.process(image)
    #brazo derecho
    veinte_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x)
    veinte_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y)
      
    dieciocho_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x)
    dieciocho_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y)

    veintidos_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x)
    veintidos_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y)  

    dieciseis_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)
    dieciseis_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y)

    catorce_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x)
    catorce_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y)

    doce_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
    doce_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)

    #Torso del lado derech
    veinticuatro_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x)
    veinticuatro_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y)
        
    veintiseis_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x)
    veintiseis_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)

    #pie lado derecho
    veintiocho_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x)
    veintiocho_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

    treinta_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x)
    treinta_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y)

    treintaydos_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x)
    treintaydos_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y)

    #brazo izquierdo
    diecinueve_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x)
    diecinueve_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y)

    diecisiete_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x)
    diecisiete_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y)

    veintiuno_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x)
    veintiuno_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y)

    quince_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
    quince_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y)

    trece_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x)
    trece_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y)

    once_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x)
    once_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        
    #Torso del lado izquierdo
    veintitres_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x)
    veintitres_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y)

    veinticinco_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x)
    veinticinco_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)

    #pie lado izquierdo
    veintisiete_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x)
    veintisiete_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y)

    veintinueve_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x)
    veintinueve_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y)

    treintayuno_x=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x)
    treintayuno_y=(poseResult.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)
    
    print('-------------GENERANDO PUNTOS-----------------')

    #Determinar vectores dependiendo de la pose 
    dieciseis_veintidos= np.array([dieciseis_x, dieciseis_y]) - np.array([veintidos_x, veintidos_y])
    np.save('16_22', dieciseis_veintidos)
    # print(dieciseis_veintidos)
    
    dieciseis_veinte=np.array([dieciseis_x, dieciseis_y]) - np.array([veinte_x, veinte_y])
    np.save('16_20', dieciseis_veinte)
    # print(dieciseis_veinte)
    
    dieciseis_dieciocho=np.array([dieciseis_x, dieciseis_y]) - np.array([dieciocho_x, dieciocho_y])
    np.save('16_18', dieciseis_dieciocho)
    # print(dieciseis_dieciocho)
    
    dieciseis_catorce=np.array([dieciseis_x, dieciseis_y]) - np.array([catorce_x, catorce_y])
    np.save('16_14', dieciseis_catorce)
    # print(dieciseis_catorce)
    
    catorce_doce=np.array([catorce_x, catorce_y]) - np.array([doce_x, doce_y])
    np.save('14_12', catorce_doce)
    # print(catorce_doce)
    
    quince_veintiuno=np.array([quince_x, quince_y]) - np.array([veintiuno_x, veintiuno_y])
    np.save('15_21', quince_veintiuno)
    # print(quince_veintiuno)
    
    quince_diecinueve=np.array([quince_x, quince_y]) - np.array([diecinueve_x, diecinueve_y])
    np.save('15_19', quince_diecinueve)
    # print(quince_diecinueve)
    
    quince_diecisiete=np.array([quince_x, quince_y]) - np.array([diecisiete_x, diecisiete_y]) 
    np.save('15_17', quince_diecisiete)
    # print(quince_diecisiete)
    
    quince_trece=np.array([quince_x, quince_y]) - np.array([trece_x, trece_y])
    np.save('15_13', quince_trece)
    # print(quince_trece)
    
    trece_once=np.array([trece_x, trece_y]) - np.array([once_x, once_y])
    np.save('13_11', trece_once)
    # print(trece_once)
    
    #VECTORES DEL TORSO 
    #derecho      
    doce_veinticuatro=np.array([doce_x, doce_y]) - np.array([veinticuatro_x, veinticuatro_y])
    np.save('12_24', doce_veinticuatro)
    # print(doce_veinticuatro)
    
    veinticuatro_veintiseis=np.array([veinticuatro_x, veinticuatro_y]) - np.array([veintiseis_x, veintiseis_y])
    np.save('24_26', veinticuatro_veintiseis)
    # print(veinticuatro_veintiseis)
    
    veintiseis_veintiocho=np.array([veintiseis_x, veintiseis_y]) - np.array([veintiocho_x, veintiocho_y])
    np.save('26_28', veintiseis_veintiocho)
    # print(veintiseis_veintiocho)
    
    veintiocho_treinta=np.array([veintiocho_x, veintiocho_y]) - np.array([treinta_x, treinta_y])
    np.save('28_30', veintiocho_treinta)
    # print(veintiocho_treinta)
    
    veintiocho_treintaydos=np.array([veintiocho_x, veintiocho_y]) - np.array([treintaydos_x, treintaydos_y])
    np.save('28_32', veintiocho_treintaydos)
    # print(veintiocho_treintaydos)
    
    #izquierdo
    once_veintitres=np.array([once_x, once_y]) - np.array([veintitres_x, veintitres_y])
    np.save('11_23', once_veintitres)
    # print(once_veintitres)
    
    veintitres_veinticinco=np.array([veintitres_x, veintitres_y]) - np.array([veinticinco_x, veinticinco_y])
    np.save('23_25', veintitres_veinticinco)
    # print(veintitres_veinticinco)
    
    veinticinco_veintisiete=np.array([veinticinco_x, veinticinco_y]) - np.array([veintisiete_x, veintisiete_y])
    np.save('25_27', veinticinco_veintisiete)
    # print(veinticinco_veintisiete)
    
    veintisiete_veintinueve=np.array([veintisiete_x, veintisiete_y]) - np.array([veintinueve_x, veintinueve_y])
    np.save('27_29', veintisiete_veintinueve)
    # print(veintisiete_veintinueve)
    
    veintisiete_treintayuno=np.array([veintisiete_x, veintisiete_y]) - np.array([treintayuno_x, treintayuno_y])
    np.save('27_31', veintisiete_treintayuno)
    # print(veintisiete_treintayuno)
    

    matrix_p1=np.array([dieciseis_veintidos,
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
    np.save('pose1_complete', matrix_p1)


    print(matrix_p1)
        
    
    