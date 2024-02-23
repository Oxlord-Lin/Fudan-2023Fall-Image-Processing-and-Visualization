"""
实现线性插值算法,读出一幅图像，
利用线性插值把图片空间分辨率放大N倍，然后保存图片
"""

from math import *
import numpy as np
from PIL import Image


def main():
    myPicture = '86版西游记模糊剧照.png'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im)
    N = int(input('请输入图片空间放大倍数N（要求为大于1的正整数）：'))
    pixels2 = np.empty((N*pixels.shape[0],N*pixels.shape[1]))

    for i in range(pixels.shape[0]): # 先放入顶点位置的值
        for j in range(pixels.shape[1]):
            pixels2[N*i,N*j] = pixels[i,j]

    for i in range(pixels.shape[0]): # 在第一个方向上进行线性插值
        for j in range(pixels.shape[1] - 1):
            Ni = N*i
            Nj = N*j
            for k in range(1,N):
                pixels2[Ni,Nj+k] = ((N-k)/N)*pixels[i,j] + (k/N)*pixels[i,j+1]
    
    for i in range(pixels.shape[0]-1): # 在第二个方向上进行插值
        Ni = N*i
        for j in range(pixels.shape[1]-1):
            Nj = N*j
            for p in range(1,N):
                for q in range(N+1):
                    pixels2[Ni+p,Nj+q] = ((N-p)/N)*pixels2[Ni,Nj+q] + (p/N)*pixels2[Ni+N,Nj+q]

    pixels2 = pixels2.astype(np.uint8)
    new_im = Image.fromarray(pixels2)
    new_im.save('exercise_3_线性插值_'+str(myPicture)+'_N='+str(N)+'.png')


if __name__ == '__main__':
    main()