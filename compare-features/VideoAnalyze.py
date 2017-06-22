# coding: utf-8

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import json
import shutil

IMG_SIZE = (180, 320)

class VideoAnalyze():
    def __init__(self,video_name):
        #分類器の指定
        self.cascade = cv2.CascadeClassifier('cascade.xml')
        self.video_name=video_name

    def show(self, match):
        img1=match['img1']
        img2=match['img2']
        plt.figure(figsize=(16,12))
        # 左
        plt.subplot(2,2,1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        # 右
        plt.subplot(2,2,2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.show()

    def compare_frame(self, frame):
        # 特徴量の検出と特徴量ベクトルの計算
        self.top_match={}
        gray_tmp=frame.copy()
        gray_tmp = cv2.resize(gray_tmp, IMG_SIZE)
        gray_tmp=gray_tmp[10:320, 0:180]
        y=310
        x=180
        target_hist1 = cv2.calcHist([gray_tmp[0:int(y/2), 0:int(x/2)]], [0], None, [256], [0, 256])
        target_hist2 = cv2.calcHist([gray_tmp[0:int(y/2), int(x/2):x]], [0], None, [256], [0, 256])
        target_hist3 = cv2.calcHist([gray_tmp[int(y/2):y, 0:int(x/2)]], [0], None, [256], [0, 256])
        target_hist4 = cv2.calcHist([gray_tmp[int(y/2):y, int(x/2):x]], [0], None, [256], [0, 256])
        for path in glob.glob('./base_img/*.png'):
            try:
                ret=[]
                ret.append(cv2.compareHist(target_hist1, self.d_dict[path]['hist1'], cv2.HISTCMP_CHISQR_ALT))
                ret.append(cv2.compareHist(target_hist2, self.d_dict[path]['hist2'], cv2.HISTCMP_CHISQR_ALT))
                ret.append(cv2.compareHist(target_hist3, self.d_dict[path]['hist3'], cv2.HISTCMP_CHISQR_ALT))
                ret.append(cv2.compareHist(target_hist4, self.d_dict[path]['hist4'], cv2.HISTCMP_CHISQR_ALT))
#                 print("{0} : {1} : {2}".format(path, frame_path,sum(ret)))
                if not self.top_match or self.top_match['ret'] > sum(ret):
                    self.top_match['ret']=sum(ret)
                    self.top_match['path']=path
                    self.top_match['ts']={'img1':self.d_dict[path]['img'],'img2':gray_tmp}
            except cv2.error:
                ret = 100000
#        print(self.top_match['path'], self.top_match['ret'])
#        self.show(self.top_match['ts'])

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
            gray_img = cv2.imread(img_path)
            gray_img = cv2.resize(gray_img, IMG_SIZE)
            gray_img=gray_img[10:320, 0:180]
            self.d_dict[img_path]={}
            self.d_dict[img_path]['img']=gray_img
            y=310
            x=180
            self.d_dict[img_path]['hist1'] = cv2.calcHist([gray_img[0:int(y/2), 0:int(x/2)]], [0], None, [256], [0, 256])
            self.d_dict[img_path]['hist2'] = cv2.calcHist([gray_img[0:int(y/2), int(x/2):x]], [0], None, [256], [0, 256])
            self.d_dict[img_path]['hist3'] = cv2.calcHist([gray_img[int(y/2):y, 0:int(x/2)]], [0], None, [256], [0, 256])
            self.d_dict[img_path]['hist4'] = cv2.calcHist([gray_img[int(y/2):y, int(x/2):x]], [0], None, [256], [0, 256])
        i=0
        cap = cv2.VideoCapture(self.video_name)
        
        #print("{0}: is Open {1}".format(self.video_name, cap.isOpened()))
        prev_detect=np.array([[-1,-1,-1,-1]])
        while(cap.isOpened()):
#        for frame_path in glob.glob('./cat_img/*.png'):
            ret, frame = cap.read()
#            frame = cv2.imread(frame_path)
            if frame is None:
                break
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # detectMultiScale(Mat image, MatOfRect objects, double scaleFactor, int minNeighbors, int flag, Size minSize, Size maxSize)
            detect = self.cascade.detectMultiScale(frame, 1.1, 3) #物体の検出
            if type(detect) is tuple or (detect == prev_detect).all():
                prev_detect= np.array([[-1,-1,-1,-1]]) if type(detect) is tuple else detect
                continue
#            print(detect)
            #self.compare_frame(frame, frame_path)
            self.compare_frame(frame)
            i=self.create_base_sienario(frame, detect, senario_dict, i)
            prev_detect=detect
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#         cap.release()
        return senario_dict

# テストベースシナリオ
if __name__ == '__main__':
    if os.path.isdir("./s_img"):
        shutil.rmtree("./s_img")
    os.mkdir("./s_img")

    va=VideoAnalyze('./2.mp4')
    senario_dict=va.main()
    print(senario_dict)
    f = open('{0}.json'.format("base_senario"), 'w')
    json.dump(senario_dict, f)
    f.close()
