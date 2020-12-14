#coding=utf-8
import os
import argparse
from math import *
import numpy as np
import cv2
# import PIL
from PIL import Image, ImageFont, ImageDraw
from PlateCommon import *
import copy
from angle_plate import rad, change_img_angle, remove_black


def show(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

class GenPlate:
    def __init__(self,fontCh,fontEng,NoPlates):
        self.fontC =  ImageFont.truetype(fontCh,43,0);
        self.fontE =  ImageFont.truetype(fontEng,60,0);
        self.img=np.array(Image.new("RGB", (226+23,70),(255,255,255)))
        self.bg  = cv2.resize(cv2.imread("./images/template.bmp"),(226+23+6,70));
        self.smu = cv2.imread("./images/smu2.jpg");
        self.noplates_path = [];
        for parent,parent_folder,filenames in os.walk(NoPlates):
            for filename in filenames:
                path = parent+"/"+filename;
                self.noplates_path.append(path);


    def draw(self,val):
        offset= 2

        self.img[0:70,offset+8:offset+8+23]= GenCh(self.fontC,val[0]);
        self.img[0:70,offset+8+23+6:offset+8+23+6+23]= GenCh1(self.fontE,val[1]);
        for i in range(6):
            base = offset+8+23+6+23+17 +i*23 + i*6 ;
            self.img[0:70, base  : base+23]= GenCh1(self.fontE,val[i+2]);
        return self.img

    def generate(self,text):
        if len(text) == 10:
            # 得到白底黑字的车牌
            fg = self.draw(text.decode(encoding="utf-8"));
          
            # 在兴能源背景上添加黑字
            com = copy.deepcopy(self.bg)
            for i in range(70):
                for j in range(249):
                    for c in range(3):
                        if fg[i,j,c] == 0:
                            com[i,j,c] = 0

            #com = rot(com,r(60)-30,com.shape,30);
            #com = rot(com,r(40)-20,com.shape,20);
            com = rotRandrom(com,5,(com.shape[1],com.shape[0]));
            #com = AddSmudginess(com,self.smu)

            com = tfactor(com)
            #com = random_envirment(com,self.noplates_path)
            com = AddGauss(com, 1+r(2))
            com = addNoise(com)
            return com

    def genPlateString(self,pos,val):
        plateStr = "";
        box = [0,0,0,0,0,0,0,0];
        if(pos!=-1):
            box[pos]=1;
        for unit,cpos in zip(box,xrange(len(box))):
            if unit == 1:
                plateStr += val
            else:
                if cpos == 0:
                    plateStr += chars[r(31)]
                elif cpos == 1:
                    plateStr += chars[41+r(24)]
                else:
                    plateStr += chars[31 + r(34)]

        return plateStr;

    def genBatch(self, batchSize,pos,charRange, outputPath,size):
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        for i in xrange(batchSize):
            plateStr = self.genPlateString(-1,-1)
            img = self.generate(plateStr)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,size)
            # filename = os.path.join(outputPath, str(i).zfill(4) + '.' + plateStr + ".jpg")
            filename = os.path.join(outputPath, str(i).zfill(5) + '_' + plateStr + ".jpg")
            #img = remove_black(img) # 变成测角度的车牌
            cv2.imwrite(filename, img)
            print(filename, plateStr)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_ch', default='./font/platech.ttf')
    parser.add_argument('--font_en', default='./font/platechar.ttf')
    parser.add_argument('--bg_dir', default='./NoPlates')
    parser.add_argument('--out_dir', default='./data/green_plate_angle', help='output dir')
    parser.add_argument('--make_num', default=100, type=int, help='num')
    parser.add_argument('--img_w', default=120, type=int, help='num')
    parser.add_argument('--img_h', default=32, type=int, help='num')
    return parser.parse_args()

def main(args):
    G = GenPlate(args.font_ch, args.font_en, args.bg_dir)
    G.genBatch(args.make_num,2,range(31,65), args.out_dir, (args.img_w, args.img_h))

if __name__ == '__main__':
    main(parse_args())
