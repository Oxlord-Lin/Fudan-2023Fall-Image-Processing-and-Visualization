from math import *
import numpy as np
from PIL import Image
from numba import njit

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



def global_OTSU(pixel:np.array):
    """
    功能：使用OTSU，进行全局二值化
    """
    hist = globalHistogram(pixel)
    S = int(sum(hist)) 
    one_hist = hist/S # 归一化
    mu_global =  np.inner(one_hist, np.array(range(256)))
    T = 127 # 为了防止出现纯色的图片（比如灰度全都为255）而找不出T，这里先预设一个，防止报错
    sigma = -float('inf')

    for t in range(1,255):
        # 以下是OTSU算法的核心部分，该算法具有高效的更新方式
        omega_bk = sum(one_hist[0:t]) # 累积概率
        mean_bk = np.inner(one_hist[0:t], np.array(range(t))) # accumulated mean
        sigma_new = (mu_global*omega_bk-mean_bk)**2 / (omega_bk*(1-omega_bk)) # 前景与背景的方差
        if sigma_new>sigma:
            sigma = sigma_new
            T = t

    pixel_class = np.where(pixel>T,1,0)
    return pixel_class


def OTSU(myPicture):
    print('开始运行OTSU算法')
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im)
    pixels :np.array = np.uint8(pixels)
    pixels_class = global_OTSU(pixels)

    color_dict = {0:(255,215,0), 1:(30,144,255), 2:(192,255,62),3:(132,112,255),
                  4:(250,220,220),5:(255,127,80)}
    
    im_output = Image.fromarray(pixels_class)
    im_output = im_output.convert('RGB')
    pixels_output = im_output.load()
    pixels_class = np.transpose(pixels_class)
    for i in range(im_output.size[0]):  # for every pixel:
        for j in range(im_output.size[1]):
            pixels_output[i,j] = color_dict[ int(pixels_class[i,j]!=0) ]
    im_output.save('exercise1_result/'+'OTSU.png')

