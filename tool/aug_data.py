import os
import cv2
import shutil
import numpy as np
import random


def rotate_image(image,angle,scale=1.0):
    (h,w)=image.shape[:2]
    center=(w//2,h//2)


    M=cv2.getRotationMatrix2D(center,angle,scale)
    rotated=cv2.warpAffine(image,M,(w,h))

    return rotated
def Aug_data_crop(path,new_path,crop_size):
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
            point=np.linspace(0,12,12,endpoint=False)
            #start:返回样本数据开始点
            # stop:返回样本数据结束点
            # num:生成的样本数据量，默认为50
            # endpoint：True则包含stop；False则不包含stop
            # retstep：If True, return (samples, step), where step is the spacing between samples.(即如果为True则结果会给出数据间隔)
            # dtype：输出数组类型
            # axis：0(默认)或-1

            for i in range(len(point)):
                top=int(point[i])
                for j in range(len(point)):
                    count+=1
                    left=int(point[j])
                    bottom=top+crop_size[0]
                    right=left+crop_size[1]
                    image=image_ori[top:bottom,left:right,:]

                    new_image_path=os.path.join(new_emotion_file_path,'%000d.png'%(count))
                    cv2.imwrite(new_image_path,image)

                    count+=1
                    image_flip=cv2.flip(image,1)
                    new_image_path = os.path.join(new_emotion_file_path, '%04d.png' % (count))
                    cv2.imwrite(new_image_path,image_flip)


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
            image_ori=cv2.resize(image_ori,(236,236))

            # for i in range(len(rotate_size)):
            #     count+=1
            #     angle=rotate_size[i]
            #     image_rotate=rotate_image(image_ori,angle)
            #
            #     new_image_path=os.path.join(new_emotion_file_path,'%04d.png'%(count))
            #     cv2.imwrite(new_image_path,image_rotate)
            count += 1
            new_image_path = os.path.join(new_emotion_file_path, '%04d.png' % (count))
            cv2.imwrite(new_image_path, image_ori)

def Aug_rotate(path,new_path,rotate_size):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)

    with open(path) as f:
        lines = f.readlines()  # readlines()读取整个文件，并返回列表，一行为一个元素
        images = []
        labels = []
        for i in lines:
            items = i.split()  # split()的时候，多个空格当成一个空格；split(' ')的时候，多个空格都要分割，每个空格分割出来空。
            images.append(items[0])
            labels.append(int(items[1]))
        data_size = len(labels)  # data_size表示总共有多少个数据
    # len =len(images)
    count = 0
    for i in range(data_size):
        # image_file_path=os.path.join(emotion_file_path,image_file)
        image_ori=cv2.imread(images[i])
        image_ori=cv2.resize(image_ori,(236,236))
        new_emotion_file_path = os.path.join(new_path,str(labels[i]))
        if not os.path.exists(new_emotion_file_path):
            os.mkdir(new_emotion_file_path)

        # for i in range(len(rotate_size)):
        count+=1
        #     angle=rotate_size[i]
        #     image_rotate=rotate_image(image_ori,angle)

        new_image_path=os.path.join(new_emotion_file_path,'%04d.png'%(count))
        cv2.imwrite(new_image_path,image_ori)
        # count += 1
        # new_image_path = os.path.join(new_emotion_file_path, '%04d.png' % (count))
        # cv2.imwrite(new_image_path, image_ori)


if __name__=='__main__':
    data_path=''
    new_path=''
    nnew_path = ''
    # Aug_data_crop(data_path,new_path,(236,236))
    # Aug_data_rotate(data_path,nnew_path,(-10,-5,0,5,10))





