
# coding: utf-8

# In[ ]:

import cv2
import os
#import matplotlib.pyplot as plt
import numpy as np
import sys
import shutil
import json
import glob


# In[ ]:

# def show_matching(match):
#     matched=match['matched']
#     kp1=match['kp1']
#     kp2=match['kp2']
#     img1=match['img1']
#     img2=match['img2']
#     # 対応する特徴点同士を描画
#     img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matched, None, flags=2)
# #     img = cv2.drawMatches(img1, kp1, img2, kp2, matched, None, flags=2)
#     plt.figure(figsize=(16,12))
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.show()


# In[ ]:

# 特徴量抽出
# A-KAZE検出器の生成
detector = cv2.AKAZE_create()
# Brute-Force Matcher生成
bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
IMG_SIZE = (200, 200)
drop_ratio=0.5
cascade = cv2.CascadeClassifier('cascade.xml') #分類器の指定

senario_dict={}
i=0
if not os.path.isdir("./s_img"):
    os.mkdir("./s_img")

d_dict={}
for img_path in glob.glob('./base_img/*.png'):
    gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     gray_img = cv2.resize(gray_img, IMG_SIZE)
    kp, des = detector.detectAndCompute(gray_img, None)
    d_dict[img_path]={}
    d_dict[img_path]['des']=des
    d_dict[img_path]['kp']=kp
    d_dict[img_path]['img']=gray_img
    
    
cap = cv2.VideoCapture('./1.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flag, Size minSize, Size maxSize)
    detect = cascade.detectMultiScale(frame, 1.1, 3) #物体の検出
    #print(detector)
    if type(detect) is tuple:
        continue

        
    # 特徴量の検出と特徴量ベクトルの計算
    top_match={}
    tmp1=frame.copy()
    tmp2=frame.copy()
    gray_tmp2 = cv2.cvtColor(tmp2, cv2.COLOR_BGR2GRAY)
#     gray_tmp2 = cv2.resize(gray, IMG_SIZE)
    __, target_des = detector.detectAndCompute(gray_tmp2, None)
    for path in glob.glob('./base_img/*.png'):
        try:
#             matches = bf.match(target_des, d_dict[path]['des'])
#             dist = [m.distance for m in matches]
#             matched = [m for m in matches]
#             matched = sorted(matches, key=lambda x:x.distance)
            
            matches = bf.knnMatch(target_des, d_dict[path]['des'], k=2)
            dist = []
            matched=[]
            for m, n in matches:
                if m.distance < drop_ratio * n.distance:
                    matched.append([m])
                    dist.append(m.distance)

            ret = sum(dist) / len(dist)
            
            if not top_match or top_match['ret'] > ret:
                top_match['ret']=ret
                top_match['path']=path
                top_match['ts']={'matched':matched, 'img1':d_dict[path]['img'], 'kp1':d_dict[path]['kp'],'img2':gray_tmp2, 'kp2':__,}
                
                
        except cv2.error:
            ret = 100000

#     print(top_match['path'], top_match['ret'])
    #show_matching(top_match['ts'])

    for (x, y, w, h) in detect:
        # タップライン
        height, width = tmp1.shape[:2]
        tap_line=tmp1[y:y+h, 0:width]
        tap_line_name='./s_img/tap_line_{0}.jpg'.format(i)
        cv2.imwrite(tap_line_name, tap_line)
        senario_dict[i]={}
        senario_dict[i]['line']={}
        senario_dict[i]['line']['base']=top_match['path']
        senario_dict[i]['line']['match']=tap_line_name
        senario_dict[i]['line']['is_parent']=True
        senario_dict[i]['line']['scale']=1
        
        # タップ位置
        tap_img=tmp1[y:y+h, x:x+w]
        tap_name='./s_img/tap_{0}.jpg'.format(i)
        cv2.imwrite(tap_name, tap_img)
        senario_dict[i]['tap']={}
        senario_dict[i]['tap']['base']=None
        senario_dict[i]['tap']['match']=tap_name
        senario_dict[i]['tap']['scale']=1
        i+=1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# print(senario_dict)

# json.dump(senario_dict, f)

# cap.release()
# cv2.destroyAllWindows()


# In[ ]:

f = open('{0}.json'.format("sample_senario"), 'w')
json.dump(senario_dict, f)
f.close()


# In[ ]:




# In[ ]:



