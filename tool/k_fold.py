import numpy as np
from PIL import Image
import random
import cv2
import os
import shutil
from sklearn.model_selection import KFold


def cut_flip_img(img, crop_size, index):
    if index == 1:
        img = cv2.flip(img, 1)

    crop = random.choice([[0, 0], [0, 12], [12, 0], [12, 12], [6, 6]])
    top = crop[0]
    left = crop[1]
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    cut_img = img[top:bottom, left:right, :]

    return cut_img

def rotate_img(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def Aug_data(path,new_path):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)
    angles = [-10, -5, 0, 5, 10]

    emotions = sorted(os.listdir(path))
    for emotion in emotions:
        emotion_path = os.path.join(path, emotion)

        new_emotion = os.path.join(new_path, emotion)
        if not os.path.exists(new_emotion):
            os.mkdir(new_emotion)

        actors = sorted(os.listdir(emotion_path))
        for actor in actors:
            actor_path = os.path.join(emotion_path, actor)

            new_actor = os.path.join(new_emotion, actor)
            if not os.path.exists(new_actor):
                os.mkdir(new_actor)

            aug_count = 0
            for angle in angles:
                aug_count += 1
                one_aug = os.path.join(new_actor, '%d' % aug_count)
                if os.path.exists(one_aug):
                    shutil.rmtree(one_aug)
                os.mkdir(one_aug)

                frames = sorted(os.listdir(actor_path))
                for frame in frames:
                    frame_path = os.path.join(actor_path, frame)
                    img = cv2.imread(frame_path)
                    rotate_frame = rotate_img(img, angle)
                    print (rotate_frame.shape)
                    new_aug_path = os.path.join(one_aug, frame)
                    cv2.imwrite(new_aug_path, rotate_frame)





def k_fold_txt(data_path,fold_path,data_name,k,if_aug):
    if os.path.exists(fold_path):
        shutil.rmtree(fold_path)
    os.mkdir(fold_path)

    kf = KFold(n_splits=k)
    count = 0
    emotions = sorted(os.listdir(data_path))
    all_id = []
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        actors = sorted(os.listdir(emotion_path))

        for actor in actors:
            if data_name == 'ck' or data_name == 'oulu':
                actor_id = actor
                all_id.append(actor_id)

            if data_name == 'mmi':
                actor_id = actor.split('-')[0]
                all_id.append(actor_id)

    all_id = sorted(list(set(all_id)))
    for train, test in kf.split(all_id):
        count += 1
        train_fold_name = '%d' % count + '_train.txt'
        train_fold_path = os.path.join(fold_path, train_fold_name)
        f_train = open(train_fold_path, 'w')

        test_fold_name = '%d' % count + '_test.txt'
        test_fold_path = os.path.join(fold_path, test_fold_name)
        f_test = open(test_fold_path, 'w')

        train_id = [all_id[index] for index in train]
        print(train_id)
        test_id = [all_id[index] for index in test]
        print(test_id)

        for emotion in emotions:
            emotion_path = os.path.join(data_path, emotion)
            actors = sorted(os.listdir(emotion_path))
            actor_id = ''
            for actor in actors:
                if data_name == 'ck' or data_name == 'oulu':
                    actor_id = actor
                if data_name == 'mmi':
                    actor_id = actor.split('-')[0]

                if actor_id in train_id:
                    actor_path = os.path.join(emotion_path, actor)

                    if if_aug:
                        augs = sorted(os.listdir(actor_path))
                        for aug in augs:
                            aug_path = os.path.join(actor_path, aug)
                            aug_imgs = sorted(os.listdir(aug_path))
                            for aug_img in aug_imgs:
                                aug_img_path = os.path.join(aug_path, aug_img)
                                label = emotion
                                path_label = aug_img_path + ' ' + '%s' % label + '\n'
                                f_train.writelines(path_label)

                    else:
                        path_label = actor_path + ' ' + '%s' % emotion + '\n'
                        f_train.writelines(path_label)

                if actor_id in test_id:
                    actor_path = os.path.join(emotion_path, actor)

                    if if_aug:
                        augs = sorted(os.listdir(actor_path))
                        for aug in augs[2]:
                            aug_path = os.path.join(actor_path, aug)
                            aug_imgs = sorted(os.listdir(aug_path))
                            for aug_img in aug_imgs:
                                aug_img_path = os.path.join(aug_path, aug_img)
                                label = emotion
                                path_label = aug_img_path + ' ' + '%s' % label + '\n'
                                f_test.writelines(path_label)

                    else:
                        path_label = actor_path + ' ' + '%s' % emotion + '\n'
                        f_test.writelines(path_label)



"""data_path='/media/zsl/D/1emotion_data/iccv/unaug/236_gray/mmi'
aug_path='/media/zsl/D/1emotion_data/iccv/unaug/236_gray/mmi_aug'
Aug_data(data_path,aug_path)"""




path='/media/zsl/D/1emotion_data/iccv/aug/236_gray/mmi_aug'
fold_path='/media/zsl/D/1emotion_data/iccv/aug/236_gray/mmi_txt'
k_fold_txt(path,fold_path,data_name='mmi',k=10,if_aug=True)




