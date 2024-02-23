import numpy as np
from PIL import Image
from numba import njit


# 腐蚀函数
@njit
def erode(pixels:np.array,b):
    """
    para: pixels是图像的灰度值矩阵
    b: b是结构元的大小，按照chessboard距离进行衡量；结构元为正方形
    return：腐蚀后的图像的灰度值矩阵
    """
    new_pixels = np.zeros(pixels.shape)
    height, width = pixels.shape
    k = b//2
    for i in range(height):
        for j in range(width):
            up = max(0,i-k)
            down = min(height,i+k)
            left = max(0,j-k)
            right = min(width,j+k)
            new_pixels[i,j] = np.max(pixels[up:down,left:right])
    return new_pixels


# 膨胀函数
@njit
def dialte(pixels:np.array,b):
    """
    para: pixels是图像的灰度值矩阵
    b: b是结构元的大小，按照chessboard距离进行衡量；结构元为正方形
    return：膨胀后的图像的灰度值矩阵
    """
    new_pixels = np.zeros(pixels.shape)
    height, width = pixels.shape
    k = b//2
    for i in range(height):
        for j in range(width):
            up = max(0,i-k)
            down = min(height,i+k)
            left = max(0,j-k)
            right = min(width,j+k)
            new_pixels[i,j] = np.min(pixels[up:down,left:right])
    return new_pixels


# 闭操作，用于填补孔洞
@njit
def closing(pixels:np.array,b):
    """
    para: pixels是图像的灰度值矩阵
    b: b是结构元的大小，按照chessboard距离进行衡量；结构元为正方形
    return：进行“闭”操作（先膨胀，后腐蚀）的图像的灰度值矩阵
    """
    pixels = dialte(pixels,b) # 先膨胀
    pixels = erode(pixels,b) # 后腐蚀
    return pixels


# 开操作，用于消除噪声
@njit
def opening(pixels:np.array,b):
    """
    para: pixels是图像的灰度值矩阵
    b: b是结构元的大小，按照chessboard距离进行衡量；结构元为正方形
    return：进行“开”操作（先腐蚀，后膨胀）的图像的灰度值矩阵
    """
    pixels = erode(pixels,b) #先腐蚀
    pixels = dialte(pixels,b) # 后膨胀
    return pixels


def main():
    # 首先导入图像，并且转换为numpy的array类型
    myPicture = 'zmic_fdu_noise.bmp'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im)
    
    k1 = 7 # 闭操作的结构元的大小
    k2 = 5 # 开操作的结构元的大小

    # 先对图像进行“闭操作”，填补孔洞
    pixels = closing(pixels,k1) 
    # 再对图像进行“开操作”，消除噪声
    pixels = opening(pixels,k2) 

   
    pixels = np.uint8(pixels)  # 转成uint8格式，否则无法正常保存为图像
    im = Image.fromarray(pixels) 
    im.save('exercise2_result/'+'result.png') # 存储图像


if __name__ == '__main__':
    main()

