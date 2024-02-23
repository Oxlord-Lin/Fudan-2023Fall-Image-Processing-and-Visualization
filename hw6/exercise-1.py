import numpy as np
import random
from PIL import Image
import copy
from numba import njit


# 计算全局的灰度直方图
@njit
def globalHistogram(pixels:np.array) -> np.array:
    """
    para:pixels图片的灰度值矩阵
    return:全局灰度直方图
    """
    hist = np.zeros(256)
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            p = pixels[i,j]
            hist[p] += 1
    return hist


# Kmeans分割
def Kmeans(pixels:np.uint8, k:int) -> np.array:
    """
    para: pixels 是原图的像素灰度矩阵
    para: k 表示要分成几类
    return: 一个与原图相同尺寸的矩阵，矩阵的每个元素记录了该像素属于哪一个类别
    """
    hist = globalHistogram(pixels) # 原图的全局灰度直方图，kmeans本质上是对全局灰度直方图进行聚类
    classification = np.uint8(np.zeros(256)) # 用于存储每个灰度值的类别
    epsilon = 0.1 # 迭代停止阈值
    cla = np.empty(pixels.shape) # 用于记录最优的分割结果
    W = float('inf') # 损失函数，即类内离差平方和

    for _ in range(5):   # 由于kmeans算法受到初始值的影响，可能会陷入局部最优，因此需要多次尝试
        # 随机初始化灰度值的类中心
        centers = np.float64(sorted(random.sample(range(256),k)))
        while True: 
            # 第一步，更新灰度值的类别
            for p in range(256):
                classification[p] = np.argmin([(p-center)**2 for center in centers]) # 分到最近的center
            
            # 第二步，更新聚类中心
            flg = 0 # 用于标识是否出现了某一类没有像素的情况
            new_centers = np.float64(np.zeros(k))
            for index in range(len(new_centers)): # index表示是哪一个中心
                subHist = np.where(classification==index,1,0)*hist # 将所有属于该类的灰度值的直方图选取出来
                # 如果有一个类没有属于它的点了，则重新选点，重新开始
                if np.sum(subHist) < 1: 
                    flg = 1
                    continue # 跳出内部的for循环
                else: # 聚类中心更新
                    new_centers[index] = np.inner(subHist,np.array(range(256))) / np.sum(subHist)
            
            if flg == 1: # 如果某个类不含任何像素，则重新随机选中心，重新开始迭代
                print('某一个类不包含任何像素，重新随机选取聚类中心，重新开始迭代')
                centers = sorted(random.sample(range(256),k))
                continue

            # 根据新、旧聚类中心（可以各自表示为k维向量）之间的距离变化，判断是否满足迭代终止条件
            if np.linalg.norm(centers-new_centers) < epsilon:
                centers = new_centers
                break
            else:
                centers = new_centers

        # 计算损失函数
        W_new = 0
        for p in range(256):
            classification[p] = np.argmin([(p-center)**2 for center in centers]) # 分到最近的center
            # 更新损失函数W
            W_new += hist[p]*(p-centers[classification[p]])**2
        if W_new < W:
            cla = copy.deepcopy(classification)
            W = W_new

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            pixels[i,j] = cla[pixels[i,j]]

    return pixels
 
 
# 一些好看的颜色
color_dict = {0:(255,215,0), 1:(30,144,255), 2:(192,255,62),
                3:(132,112,255),4:(250,220,220),5:(255,127,80)}

def main():
    # 先用OSTU算法对图像进行二值化
    from util import OTSU
    OTSU(myPicture = '受到椒盐噪声污染的肝脏图像.png')

    # 以下是kmeans
    for k in (2,3,4,5,6):
        print(k)
        
        #导入图片
        myPicture = '受到椒盐噪声污染的肝脏图像.png'
        im_raw = Image.open(myPicture)
        im = im_raw.convert('L')
        pixels = np.array(im)
        pixels = np.uint8(pixels)
        
        # 基于Kmeans进行分割
        pixels_class = Kmeans(pixels,k)
        im_output = Image.fromarray(pixels_class)
        im_output = im_output.convert('RGB')
        pixels_output = im_output.load()
        pixels_class = np.transpose(pixels_class)
        for i in range(im_output.size[0]):  # for every pixel:
            for j in range(im_output.size[1]):
                pixels_output[i,j] = color_dict[pixels_class[i,j]]
        im_output.save('exercise1_result/'+'k='+str(k)+'.png')


if __name__ == '__main__':
    main()



