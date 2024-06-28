import os
import random


# def make_txt(data_path,new_path):
#     f=open(new_path,'w')
#     all_exp=sorted(os.listdir(data_path))
#     print(all_exp)
#     for w in all_exp:
#         neww = os.path.join(data_path, w)
#         print(neww)
#         t = sorted(os.listdir(neww))
#         for x in t:
#             newx = os.path.join(neww, x)
#             print(newx)
#             q = sorted(os.listdir(newx))
#             for y in q:
#                 # newx = os.path.join(newx, y)
#                 print("y")
#                 print(y)
#                 # print(newx)
#                 emotion_label=int(w)
#                 # single_exp_path=os.path.join(newx,y)
#                 # single_exp=os.listdir(single_exp_path)
#
#                     # print(exp_img)
#                 exp_img_path=os.path.join(newx,y)
#                 exp_img_path = exp_img_path.replace('\\','/')
#                 f.writelines(exp_img_path+' '+'%d'%emotion_label+'\n')
#     f.close()

def make_txt(data_path,new_path):
    f=open(new_path,'w')
    all_exp=sorted(os.listdir(data_path))
    for exp_class in all_exp:
        emotion_label=int(exp_class)
        single_exp_path=os.path.join(data_path,exp_class)
        single_exp=os.listdir(single_exp_path)
        for exp_img in single_exp:
            exp_img_path=os.path.join(single_exp_path,exp_img)
            exp_img_path = exp_img_path.replace('\\','/')
            f.writelines(exp_img_path+' '+'%d'%emotion_label+'\n')
    f.close()
def mess_txt(data_path,new_path):
    all=[]
    f=open(data_path,'r')
    new=open(new_path,'w')
    lines=f.readlines()
    random.shuffle(lines)

    for line in lines:
        new.writelines(line)




data_path='/media/zsl/D/yjy/data/FER/fer2013_p_236/PrivateTest/'
new_path='/media/zsl/D/yjy/data/FER/fer2013_p_236/PrivateTest/fer2013.txt'
make_txt(data_path,new_path)
