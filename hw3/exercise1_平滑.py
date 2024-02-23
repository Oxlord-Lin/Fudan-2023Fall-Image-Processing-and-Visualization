"""实现图像域基于空间滤波器的平滑操作"""
import numpy as np
from numba import jit  # 加速器
import copy
from PIL import Image
from math import floor, pi, exp

@jit # 加速！
def smooth_averayge(pixels,kernelSize):
    """
    功能：取卷积核覆盖范围内所有像素点平均值进行平滑操作
    input:
        pixels：图像的所有像素点，一个矩阵
        kernelSize：卷积核的大小
    output:
        进行平滑操作之后的图像的所有像素点   
    """
    # 首先进行padding
    m = kernelSize//2 # padding时边缘增加的宽度
    P = np.ones((pixels.shape[0]+2*m,pixels.shape[1]+2*m)) 
    P = P*127 # 边缘增加的部分用127填充
    P[m:P.shape[0]-m, m:P.shape[1]-m] = pixels
    pixels = copy.deepcopy(P) # 深拷贝
    # 平滑操作
    for i in range(m,P.shape[0]-m):
        for j in range(m,P.shape[1]-m):
            pixels[i,j] = np.sum(P[i-m:i+m+1, j-m:j+m+1])/kernelSize**2
    # 只返回中间部分
    return pixels[m:P.shape[0]-m,m:P.shape[1]-m]


def smooth_Gauss(pixels,sigma):
    """
    功能：将卷积核设为高斯核，并根据标准差sigma自动调整卷积核大小（3倍标准差之外都设为0），进行平滑化处理
    input:
        pixels：图像的所有像素点，一个矩阵
        sigma：高斯核的标准差
    output:
        进行平滑操作之后的图像的所有像素点
    """
    # 首先进行padding
    # from math import floor
    kernelSize = 1+2*floor(3*sigma)
    Kernal = np.empty((kernelSize,kernelSize)) # 高斯核
    m = kernelSize//2 # padding时边缘增加的宽度
    # from math import pi, exp
    for i in range(kernelSize):
        for j in range(kernelSize):
            Kernal[i,j] = (1/(2*pi*sigma**2)) * exp(-((i-m)**2+(j-m)**2)/(2*sigma**2))
    # print(np.sum(Kernal))

    P = np.ones((pixels.shape[0]+2*m,pixels.shape[1]+2*m)) 
    P = P*127 # 边缘增加的部分用127填充
    P[m:P.shape[0]-m, m:P.shape[1]-m] = pixels
    pixels = copy.deepcopy(P) # 深拷贝
    # 平滑操作
    for i in range(m,P.shape[0]-m):
        for j in range(m,P.shape[1]-m):
            pixels[i,j] = np.sum(P[i-m:i+m+1, j-m:j+m+1]*Kernal) # 元素相乘，Hadarmard积
    # 只返回中间部分
    return pixels[m:P.shape[0]-m,m:P.shape[1]-m]

def main():
    myPicture = 'test.jpeg'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im) 


    # 以下用取局部平均值的方法进行平滑处理
    for kernelSize in [3,5,7,9]:
        pixels2 = smooth_averayge(pixels, kernelSize)
        pixels2 = pixels2.astype(np.uint8)
        new_im = Image.fromarray(pixels2)
        new_im.save('result1/smooth_average_kernelSize='+str(kernelSize)+'.png')


    # 以下用高斯核的方法进行平滑处理
    for sigma in [1,2,3,4]:
        pixels2 = smooth_Gauss(pixels, sigma)
        pixels2 = pixels2.astype(np.uint8)
        new_im = Image.fromarray(pixels2)
        new_im.save('result1/smooth_Gauss_sigma='+str(sigma)+'.png')


if __name__ == '__main__':
    main()