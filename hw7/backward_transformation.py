"""实现基于FFD的图像反变换"""

from FFD import ffd # 自由形变算法
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 可交互
import matplotlib.pyplot as plt
matplotlib.rc("font",family='YouYuan') # 显示中文字体
import cv2 as cv
from collections import defaultdict

## -----------------------------辅助函数---------------------------------
def interpolation(coord:np.array,ori_img:np.array):
    """
    线性插值函数
    给定浮点型坐标和图像，找到最邻近的四个像素进行插值，对于越界的像素点直接截断
    input:
        coord : the coordinates of the point
        ori_img: the original image
    output:
        value : the interpolation value
    """
    x , y = coord
    oh, ow = ori_img.shape

    # 对越界坐标进行截断
    if x < 0 : x = 0
    if x >= oh : x = oh-1
    if y < 0 : y = 0
    if y >= ow : y = ow-1

    # 找到距离最近的四个像素点位置
    min_x , min_y = int(x) , int(y)
    max_x , max_y = min(oh-1,min_x+1) , min(ow-1,min_y+1)

    # 线性插值
    u , v = x - min_x , y-min_y
    return (1-u)*(1-v)*ori_img[min_x,min_y] + u*(1-v)*ori_img[max_x,min_y] +\
          (1-u)*v*ori_img[min_x,max_y] + u*v*ori_img[max_x,max_y]


def backward_trans(ori_image:np.array, m:int, n:int, shift_dict:dict):
    """
    实现基于FFD的反变换
    input：
        ori_image:原始图
        m,n: 网格的高和宽，将会在【目标图】上划分出(m+1)*(n+1)的均匀等分控制点
        shift_dict: 字典，用于传入从【目标图】到【原始图】的控制点的位移，
            格式为{(i,j): delta_x, delta_y}，
            其中 (i,j) 表示第i行，第j列的控制点，
            delta_x, delta_y 分别表示在第一个维度和第二个维度上发生的位移
    output：
        goal_image: 目标图
    """
    goal_image = np.zeros(ori_image.shape)
    shifted_index = ffd(goal_image.shape[0],goal_image.shape[1],m,n,shift_dict)
    for i in range(goal_image.shape[0]):
        for j in range(goal_image.shape[1]):
            goal_image[i,j] = interpolation(shifted_index[i,j,:],ori_image)
    
    return goal_image

## -----------------------读取图像，并由用户选取控制点----------------------------

fig = plt.figure(figsize=(16,12))
ori_img = cv.imread("grid.jpg",0)
goal_img = cv.imread("grid.jpg",0)
ori_img = cv.resize(ori_img,goal_img.shape)


# 选取控制点
plt.subplot(121)
plt.imshow(ori_img,cmap=plt.get_cmap("gray"))
plt.title("该图为原始图像，请您用鼠标点击选择控制点，最多30个")
x = plt.ginput(n=30,timeout=0) # 回车结束选点
ori_list = np.float32([[int(c[1]),int(c[0])] for c in x]) # 图像显示的坐标轴和numpy的矩阵坐标轴的位置正好相反

plt.subplot(122)
plt.imshow(goal_img,cmap=plt.get_cmap("gray"))
plt.title("请用鼠标点击选择原图的控制点变换后的位置")
y = plt.ginput(n=30,timeout=0) # 回车结束选点
goal_list = np.float32([[int(c[1]),int(c[0])] for c in y]) # 图像显示的坐标轴和numpy的矩阵坐标轴的位置正好相反


## -----------------将用户指定的控制点以及控制点位移转换为字典进行存储-----------------------

def return_no_shift():
    return np.zeros(2)
shift_dict = defaultdict(return_no_shift)  # 使用默认字典存储所有发生位移的控制点，可以节省存储空间
m = 10
n = 10
lx = goal_img.shape[1]/m
ly = goal_img.shape[0]/n
for k in range(len(goal_list)):
    p1 = goal_list[k] # 目标图上的控制点
    p2 = ori_list[k]    # 原始图上的控制点
    # 将任意选定的【目标图】上的控制点，移到最近的网格点上
    ix = round(p1[0]/lx) 
    iy = round(p1[1]/ly) 
    shift_dict[(ix,iy)] = np.array([p2[0]-p1[0], p2[1]-p1[1]])


## -----------------------基于FFD进行图像反变换--------------------------------

fig = plt.figure(figsize=(16,12))
plt.subplot(121)
plt.imshow(ori_img,cmap="gray")
# plt.axis("off")
plt.title("原始图像")

new_img = backward_trans(ori_img, m, n, shift_dict)  # 基于FFD进行图像反变换

plt.subplot(122)
plt.imshow(new_img,cmap="gray")
plt.title("形变图像")
# plt.axis("off")

plt.show() # 展示图像