import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import shutil
def rotate_image(image,angle,scale=1.0):
    (h,w)=image.shape[:2]
    center=(w//2,h//2)


    M=cv2.getRotationMatrix2D(center,angle,scale)
    rotated=cv2.warpAffine(image,M,(w,h))
    return rotated
def face_aligment(path,new_path):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)

    emotion_files=sorted(os.listdir(path))
    for emotion_file in emotion_files:
        emotion_file_path = os.path.join(path, emotion_file)
        new_emotion_file_path = os.path.join(new_path, emotion_file)
        print(new_emotion_file_path)

        if not os.path.exists(new_emotion_file_path):
            os.mkdir(new_emotion_file_path)

        image_files=sorted(os.listdir(emotion_file_path))
        count = 0
        for image_file in image_files:
            count+=1
            image_file_path = os.path.join(emotion_file_path, image_file)
            # image_ori = cv2.imread(image_file_path)
            face = cv2.imread(image_file_path).astype('uint8')
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 用来预测关键点
            rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
            # print(face.shape[0], face.shape[1])
            shape = predictor(np.uint8(face), rec)  # 注意输入的必须是uint8类型
            order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
            # for j in order:
            #     x = shape.part(j).x
            #     y = shape.part(j).y
                # cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

            eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼的中心坐标
                          (shape.part(36).y + shape.part(45).y) * 1. / 2)
            dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
            dy = (shape.part(45).y - shape.part(36).y)

            angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
            RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
            RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
            RotImg = cv2.cvtColor(RotImg, cv2.COLOR_BGR2RGB)
            new_image_path = os.path.join(new_emotion_file_path, '%04d.png' % (count))
            cv2.imwrite(new_image_path, RotImg)
def Aug_data_rotate(path,new_path,rotate_size):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)

    emotion_files=sorted(os.listdir(path))
    for emotion_file in emotion_files:
        emotion_file_path=os.path.join(path,emotion_file)
        new_emotion_file_path=os.path.join(new_path,emotion_file)
        print (new_emotion_file_path)

        if not os.path.exists(new_emotion_file_path):
            os.mkdir(new_emotion_file_path)

        image_files=sorted(os.listdir(emotion_file_path))
        count=0
        for image_file in image_files:
            image_file_path=os.path.join(emotion_file_path,image_file)
            image_ori=cv2.imread(image_file_path)


            for i in range(len(rotate_size)):
                count+=1
                angle=rotate_size[i]
                image_rotate=rotate_image(image_ori,angle)

                new_image_path=os.path.join(new_emotion_file_path,'%04d.png'%(count))
                cv2.imwrite(new_image_path,image_rotate)
face_aligment('/media/zsl/D/yjy/data/kaggle/236/ALIG/Traing','/media/zsl/D/yjy/data/kaggle/236/ALIG/alig_traing')
