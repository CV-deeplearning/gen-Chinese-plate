#coding:utf8
import os
import cv2
import numpy as np
import random
import copy
import sys


def rad(x):
    return x * np.pi / 180

def change_img_angle(img, anglex, angley, anglez):
    ih, iw = img.shape[:2]
    fov = 42
    # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z = np.sqrt(ih ** 2 + iw ** 2) / 2 / np.tan(rad(fov / 2))
    # 齐次变换矩阵
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, -np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    # 四对点的生成
    pcenter = np.array([iw / 2, ih / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([ih, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, iw, 0, 0], np.float32) - pcenter
    p4 = np.array([ih, iw, 0, 0], np.float32) - pcenter
    
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]

    org = np.array([[0, 0],
                    [ih, 0],
                    [0, iw],
                    [ih, iw]], np.float32)

    dst = np.zeros((4, 2), np.float32)
    #face_point_rst = np.zeros((4, 2), np.float32)
    #plate_point_rst = np.zeros((4, 2), np.float32)

    # 投影至成像平面
    for i in range(4):
        dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
        dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)
    result = cv2.warpPerspective(img, warpR, (iw, ih))
    return result

def remove_black(img):
    height, width, _ = img.shape
    angle = random.choice([36, 38, 40, 42, 44])
    img_new = change_img_angle(img, 0, -1*angle, 0)
    img_gray = cv2.cvtColor(img_new, cv2.COLOR_RGB2GRAY)
    for j in range(int(width/2.0), width):
        pix_num = img_gray[int(height/2),j]
        if pix_num == 0:
            break
    return img_new[:, 0:j]


if __name__ == "__main__":
    img_path = "/home/gp/work/project/end-to-end-for-chinese-plate-recognition/plate/04.jpg"
    img = cv2.imread(img_path)
    img_new = remove_black(img)
    cv2.imwrite("haha.jpg", img_new)
