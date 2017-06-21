# coding: utf-8

from CompareFeatures import  CompareFeatures
from VideoAnalyze import  VideoAnalyze
import json
import sys
import os


#sys.argv=['run.py','1.mp4']


# テストベースシナリオ
if __name__ == '__main__':
    # 動画 引数必須
    if len(sys.argv) < 1:
        print('Usage: python3 video_analyze.py video_name.mp4')
        sys.exit(1)

    va=VideoAnalyze(sys.argv[1])
    try:
        senario_dict=va.main()
#         f = open('{0}.json'.format("base_senario"), 'w')
#         json.dump(senario_dict, f)
#         f.close()
    except Exception as e:
        print("ERROR : Generate to base_senario.json. ")
        print(e)
        sys.exit(1)
    
    if not os.path.isdir("./s_img"):
        os.mkdir("./s_img")
    
    # ベースシナリオjson読み込み
#     f = open('base_senario.json', 'r')
#     data = json.load(f)
#    print(senario_dict)
    data = senario_dict

    # 解像度
    scale='#540*888#'
    print(scale)
    cf = CompareFeatures()
    for key in data:
        bs=data[key]
        cf.load_imgs(bs['line']['base'],bs['line']['match'],scale=(1,bs['line']['scale']))
        cf.get_postion(IS_PARENT=True)
        cf.load_imgs(bs['tap']['base'],bs['tap']['match'],scale=(1,bs['tap']['scale']))
        try:
            pos = cf.get_postion()
            print("タップ:0,{0},{1},0,0,0".format(int(pos[0]),int(pos[1])))
            print("待機:10,10,0,0,0,0")
        except:
            pass
