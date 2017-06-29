# coding: utf-8

import cv2
import os
#import matplotlib.pyplot as plt
import numpy as np
import sys
import glob
import json

IMG_SIZE = (180, 320)

class VideoAnalyze():
    def __init__(self,video_name):
        self.video_name=video_name
        # 8近傍の定義
        self.neiborhood8 = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]],np.uint8)

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
        
    def show_matching(self, match):
        matched=match['matched']
        kp1=match['kp1']
        kp2=match['kp2']
        img1=match['img1']
        img2=match['img2']
        # 対応する特徴点同士を描画
        img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matched, None, flags=2)
        plt.figure(figsize=(16,12))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
                if not self.top_match or self.top_match['ret'] > sum(ret):
                    self.top_match['ret']=sum(ret)
                    self.top_match['path']=path
                    self.top_match['ts']={'img1':self.d_dict[path]['img'],'img2':gray_tmp}
            except cv2.error:
                ret = 100000

    def create_base_sienario(self, frame, rect, senario_dict, i):
        tmp=frame.copy()
        for (x, y, w, h) in rect:
            if y < 30 or h > 500 or w > 500:
                continue
            # タップライン
            height, width = tmp.shape[:2]
            if h < 100:
                h_buff=int((100-h)/2)                
                y = y - h_buff
                h = h + h_buff*2
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
            if w < 100:
                w_buff=int((100-w)/2)
                x-=w_buff
                w+=w_buff*2
            tap_img=tmp[y:y+h, x:x+w]
            tap_name='./s_img/tap_{0}.jpg'.format(i)
            cv2.imwrite(tap_name, tap_img)
            senario_dict[i]['tap']={}
            senario_dict[i]['tap']['base']=None
            senario_dict[i]['tap']['match']=tap_name
            senario_dict[i]['tap']['scale']=1
            i+=1
        return i

    def flame_diff(self,im1,im2,im3,th, tl,blur):
        im1=cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2=cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        im3=cv2.cvtColor(im3, cv2.COLOR_RGB2GRAY)
        mask_img=cv2.imread('./mask.png', 0)
        d1 = cv2.absdiff(im3, im2)
        d2 = cv2.absdiff(im2, im1)
        if len(mask_img):
            diff = cv2.bitwise_and(d1, d2, mask=mask_img)
        else:
            diff = cv2.bitwise_and(d1, d2)
        
        # 差分が閾値より小さければTrue
        mask = diff<th

        # 背景画像と同じサイズの配列生成
        im_mask = np.empty((im1.shape[0],im1.shape[1]),np.uint8)
        im_mask[:][:]=255
        # Trueの部分（背景）は黒塗り
        im_mask[mask]=0

        # 8近傍で膨張処理
        im_mask = cv2.dilate(im_mask,
                              self.neiborhood8,
                              iterations=20)
        # ノイズ除去
        im_mask = cv2.medianBlur(im_mask,blur)

        return  im_mask

    def find_rect(self,image):
        _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in contours:
            approx = cv2.convexHull(contour)
            rect = cv2.boundingRect(approx)
            if rect[2]>70 and rect[3]>70 and rect[2]<200 and rect[3]<200:
                rects.append(np.array(rect))
        return rects
    
    def main(self):
        senario_dict={}
        self.d_dict={}
        for img_path in glob.glob('./base_img/*.png'):
            img = cv2.imread(img_path)
            h,w = img.shape[:2]
            img = cv2.resize(img, IMG_SIZE)
            img=img[10:IMG_SIZE[1], 0:IMG_SIZE[0]]    
            self.d_dict[img_path]={}
            self.d_dict[img_path]['img']=img
            y=IMG_SIZE[1] - 10
            x=IMG_SIZE[0]
            self.d_dict[img_path]['hist1'] = cv2.calcHist([img[0:int(y/2), 0:int(x/2)]], [0], None, [256], [0, 256])
            self.d_dict[img_path]['hist2'] = cv2.calcHist([img[0:int(y/2), int(x/2):x]], [0], None, [256], [0, 256])
            self.d_dict[img_path]['hist3'] = cv2.calcHist([img[int(y/2):y, 0:int(x/2)]], [0], None, [256], [0, 256])
            self.d_dict[img_path]['hist4'] = cv2.calcHist([img[int(y/2):y, int(x/2):x]], [0], None, [256], [0, 256])
        scale=(w,h)
        i=0
        ct=0
        is_print=True
        cap = cv2.VideoCapture(self.video_name)
        _, frame1 = cap.read()
        _, frame2 = cap.read()
        ret, frame3 = cap.read()
        while ret:
            frame_fs = self.flame_diff(frame1.copy(),frame2.copy(),frame3.copy(), 5, 0,5)
            rect = self.find_rect(frame_fs.copy())
            ct+=1

            if is_print and len(rect) and len(rect) <3:
                is_print=False
                ct=0

                # ラベルの個数nと各ラベルの重心座標cogを取得
                # label = cv2.connectedComponentsWithStats(frame_fs)
                # n = label[0] - 1
                # cog = np.delete(label[3], 0, 0)
                # test2 = cv2.circle(frame2.copy(),(int(cog[:,0].mean()),int(cog[:,1].mean())), 40, (0,0,0), -1)
                # cv2.imwrite("./test/test{0:0>3}.jpg".format(i) ,test2)

                self.compare_frame(frame2)
                i=self.create_base_sienario(frame2, rect, senario_dict, i)
            elif not len(rect):
                if ct>15:
                    is_print = True

            frame1 = frame2
            frame2 = frame3
            ret, frame3 = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        return senario_dict,scale

if __name__ == '__main__':
    va=VideoAnalyze(sys.argv[1])
    senario_dict,scale=va.main()
    print(senario_dict)
    f = open('{0}.json'.format("base_senario"), 'w')
    json.dump(senario_dict, f)
    f.close()
