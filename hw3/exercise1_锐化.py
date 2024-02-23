"""实现图像域基于空间滤波器的锐化操作"""

import numpy as np
from numba import jit
from PIL import Image
from math import floor, pi, exp

@jit # 加速！
def Sharpen_Laplacian(pixels,w):
    """
    功能：将卷积核设为拉普拉斯算子，并基于拉普拉斯运算后的结果对图像进行锐化
    input:
        pixels：图像的所有像素点，一个矩阵
        w：拉普拉斯算子的系数
    output:
        进行锐化操作之后的图像的所有像素点    
    """
    # 构建拉普拉斯算子
    Kernal = np.zeros((3,3)) 
    Kernal[1,1] = -4
    Kernal[0,1] = 1
    Kernal[1,0] = 1
    Kernal[1,2] = 1
    Kernal[2,1] = 1

    # padding
    m = 1 # padding时边缘增加的宽度
    P = np.ones((pixels.shape[0]+2*m,pixels.shape[1]+2*m)) 
    P = P*127 # 边缘增加的部分用127填充
    P[m:P.shape[0]-m, m:P.shape[1]-m] = pixels
    Laplace = np.array(P) # 深拷贝，用于存储图像梯度值
    # 求出图像梯度值
    for i in range(m,P.shape[0]-m):
        for j in range(m,P.shape[1]-m):
            Laplace[i,j] = np.sum(P[i-m:i+m+1, j-m:j+m+1]*Kernal) # 元素相乘，Hadarmard积
    
    # 只要图像梯度中间的部分
    Laplace = Laplace[m:P.shape[0]-m, m:P.shape[1]-m]
    
    # 归一化
    Laplace = Laplace/np.max(Laplace) 
    # 锐化
    pixels = pixels + w*Laplace

    # 检查灰度值是否超出[0,255]，超出的部分都变成0或者255
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            p = pixels[i,j]
            if p < 0:
                pixels[i,j] = 0
            elif p > 255:
                pixels[i,j] = 255

    return pixels



def sharpen_highboost(pixels,sigma,k):
    """
    功能：先根据标准差sigma调整高斯核大小，基于高斯核做卷积求出局部范围内的灰度平均值，
        然后求出unsharp masking，最后将原图像的灰度加上unsharp masking函数值，
        完成对图像的锐化
    input:
        pixels：图像的所有像素点，一个矩阵
        sigma：求平均值时所用高斯核的标准差
        k：unsharp masking的系数
    output:
        进行锐化操作之后的图像的所有像素点    
    """
    # 首先构建高斯核用于求平均值
    # from math import floor
    kernelSize = 1 + 2*floor(3*sigma) # 高斯核的大小
    Kernal = np.empty((kernelSize,kernelSize)) # 高斯核
    m = kernelSize//2 # padding时边缘增加的宽度
    # from math import pi, exp
    for i in range(kernelSize):
        for j in range(kernelSize):
            Kernal[i,j] = (1/(2*pi*sigma**2)) * exp(-((i-m)**2+(j-m)**2)/(2*sigma**2))
    
    # # only for test
    # Kernal = np.ones((kernelSize,kernelSize))
    # Kernal = Kernal/np.sum(Kernal)

    # padding
    P = np.ones((pixels.shape[0]+2*m,pixels.shape[1]+2*m)) 
    P = P*127 # 边缘增加的部分用127填充
    P[m:P.shape[0]-m, m:P.shape[1]-m] = pixels
    unsharp_mask = np.array(P) # 深拷贝
    
    # 求平均值，并计算unsharp_mask
    for i in range(m,P.shape[0]-m):
        for j in range(m,P.shape[1]-m):
            unsharp_mask[i,j] = P[i,j] - np.sum(P[i-m:i+m+1, j-m:j+m+1]*Kernal)
    
    # 只要中间部分
    unsharp_mask = unsharp_mask[m:P.shape[0]-m,m:P.shape[1]-m]
    # 归一化
    unsharp_mask = unsharp_mask/np.max(unsharp_mask)
    # 锐化
    pixels = pixels + k*unsharp_mask

    # 检查灰度值是否超出[0,255]，超出的部分都变成0或者255
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            p = pixels[i,j]
            if p < 0:
                pixels[i,j] = 0
            elif p > 255:
                pixels[i,j] = 255

    return pixels


def main():
    myPicture = 'test2.png'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im) 

    ## 以下基于拉普拉斯算子进行锐化
    for w in [-20,-50,-80,-110]:
        pixels2 = Sharpen_Laplacian(pixels, w)
        pixels2 = pixels2.astype(np.uint8)
        new_im = Image.fromarray(pixels2)
        new_im.save('result2/sharpen_Laplacian_w='+str(w)+'.png')
    
    ## 以下基于highboost的方法进行锐化
    sigma = 1 # 高斯核的标准差
    for k in [20,50,80,110]:
        pixels2 = sharpen_highboost(pixels, sigma, k)
        pixels2 = pixels2.astype(np.uint8)
        new_im = Image.fromarray(pixels2)
        new_im.save('result2/sharpen_highboost_sigma='+str(sigma)+'_k='+str(k)+'.png')


if __name__ == '__main__':
    main()