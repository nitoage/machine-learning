# coding: utf-8

import cv2
import os
#import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

IMG_SIZE = (200, 200)
drop_ratio=0.5

class VideoAnalyze():
    def __init__(self,video_name):
        # 特徴量抽出
        # A-KAZE検出器の生成
        self.detector = cv2.AKAZE_create()
        # Brute-Force Matcher生成
        self.bf=cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)
        self.cascade = cv2.CascadeClassifier('cascade.xml') #分類器の指定
        self.video_name=video_name

    def show_matching(self, match):
        matched=match['matched']
        kp1=match['kp1']
        kp2=match['kp2']
        img1=match['img1']
        img2=match['img2']
        # 対応する特徴点同士を描画
        img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matched, None, flags=2)
#     img = cv2.drawMatches(img1, kp1, img2, kp2, matched, None, flags=2)
        plt.figure(figsize=(16,12))
    #   plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def compare_frame(self, frame):
        # 特徴量の検出と特徴量ベクトルの計算
        self.top_match={}
        tmp=frame.copy()
        gray_tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    #     gray_tmp = cv2.resize(gray, IMG_SIZE)
        __, target_des = self.detector.detectAndCompute(gray_tmp, None)
        for path in glob.glob('./base_img/*.png'):
            try:
    #             matches = bf.match(target_des, d_dict[path]['des'])
    #             dist = [m.distance for m in matches]
    #             matched = [m for m in matches]
    #             matched = sorted(matches, key=lambda x:x.distance)

                matches = self.bf.knnMatch(target_des, self.d_dict[path]['des'], k=2)
                dist = []
                matched=[]
                for m, n in matches:
                    if m.distance < drop_ratio * n.distance:
                        matched.append([m])
                        dist.append(m.distance)

                ret = sum(dist) / len(dist)

                if not self.top_match or self.top_match['ret'] > ret:
                    self.top_match['ret']=ret
                    self.top_match['path']=path
                    self.top_match['ts']={'matched':matched, 'img1':self.d_dict[path]['img'], \
                                          'kp1':self.d_dict[path]['kp'],'img2':gray_tmp, 'kp2':__,} 
            except cv2.error:
                ret = 100000

    #     print(self.top_match['path'], self.top_match['ret'])
        #show_matching(self.top_match['ts'])        

    def create_base_sienario(self, frame, detect, senario_dict, i):
        tmp=frame.copy()
        for (x, y, w, h) in detect:
            # タップライン
            height, width = tmp.shape[:2]
            tap_line=tmp[y:y+h, 0:width]
            tap_line_name='./s_img/tap_line_{0}.jpg'.format(i)
            cv2.imwrite(tap_line_name, tap_line)
            senario_dict[i]={}
            senario_dict[i]['line']={}
            senario_dict[i]['line']['base']=self.top_match['path']
            senario_dict[i]['line']['match']=tap_line_name
            senario_dict[i]['line']['is_parent']=True
            senario_dict[i]['line']['scale']=1

            # タップ位置
            tap_img=tmp[y:y+h, x:x+w]
            tap_name='./s_img/tap_{0}.jpg'.format(i)
            cv2.imwrite(tap_name, tap_img)
            senario_dict[i]['tap']={}
            senario_dict[i]['tap']['base']=None
            senario_dict[i]['tap']['match']=tap_name
            senario_dict[i]['tap']['scale']=1
            i+=1
        return i

    def main(self):
        senario_dict={}
        self.d_dict={}
        for img_path in glob.glob('./base_img/*.png'):
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #     gray_img = cv2.resize(gray_img, IMG_SIZE)
            kp, des = self.detector.detectAndCompute(gray_img, None)
            self.d_dict[img_path]={}
            self.d_dict[img_path]['des']=des
            self.d_dict[img_path]['kp']=kp
            self.d_dict[img_path]['img']=gray_img

        i=0
        cap = cv2.VideoCapture(self.video_name)
        prev_detect= np.array([[-1,-1,-1,-1]])
        #print("{0}: is Open {1}".format(self.video_name, cap.isOpened()))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flag, Size minSize, Size maxSize)
            detect = self.cascade.detectMultiScale(frame, 1.1, 3) #物体の検出
            if type(detect) is tuple or (detect==prev_detect).all():
                prev_detect= np.array([[-1,-1,-1,-1]]) if type(detect) is tuple else detect
                continue
            #print(detect)
            self.compare_frame(frame)
            i=self.create_base_sienario(frame, detect, senario_dict, i)
            prev_detect=detect
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # print(senario_dict)
        cap.release()
        return senario_dict

# テストベースシナリオ
if __name__ == '__main__':
    va=VideoAnalyze('./1.mp4')
    senario_dict=va.main()
    print(senario_dict)
