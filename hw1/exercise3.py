"""Implement the algorithm of local histogram equalization: 
(1) first implement histogram equalization algorithm, 
and then (2) implement the local histogram equalization 
using efficient computation of local histogram. 
Please test your code on images and show the results in your report."""

from math import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import exercise2_2

def local_hist(pixels,w):
    """
    input:
        pixels is the intensity of the picture
        w is the size of the local patch, required to be an odd number
        bins is the number of partitions, usually taking 2^p
    output:
        hist_list: the first two elements of each row is the index of the central pixel,
            and the rest elements of each row are the local histogram
    """ 
    count = 0
    hist = np.zeros((pixels.size, 2 + 256)) # pixels.size返回的是pixels中有多少元素
    for i in range(pixels.shape[0]):
        up = max(0,i-w//2)
        down = min(pixels.shape[0]-1,i+w//2)
        for j in range(pixels.shape[1]):
            if j==0: # 第一个元素，重新计算patch
                hist[count,0] = i
                hist[count,1] = j
                rt = min(pixels.shape[1]-1,j+w//2) # 边界检查，防止越界
                for p in range(up,down+1):
                    for q in range(rt+1):
                        hist[count,2+pixels[p,q]] += 1
                count += 1
            else: # 其他非首位元素的local histogram只需要做列更新
                hist[count,:] = hist[count-1,:]
                hist[count,0] = i
                hist[count,1] = j
                lf = max(0,j-w//2)
                rt = min(pixels.shape[1]-1,j+w//2)
                if lf==0: # 边界检查，防止越界
                    old_col = []  
                else:
                    old_col = pixels[up:down+1,lf-1]
                if j+w//2 > rt: # 边界检查，防止越界
                    new_col = []
                else:
                    new_col = pixels[up:down+1,rt]
                for item in old_col:
                    hist[count,2+item] -= 1
                for item in new_col:
                    hist[count,2+item] += 1
                count = count + 1
    return hist 


def global_hist(pixels):
    """
    input: a 2-dimension array containg the intensity of each pixel
    output: a global histogram
    
    """
    hist = [0]*256
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            hist[pixels[i,j]] += 1
    return hist


def trans_function(hist):
    """
    input: a histogram in [0,255]
    output: a list of length 256, mapping the intensity of the original picture
        to that after equalization
    """
    cum_list = np.empty(256)
    cum_list[0] = hist[0]
    for i in range(1,256):
        cum_list[i] = cum_list[i-1] + hist[i]
    cum_list = cum_list/cum_list[-1]
    trans = np.zeros(256)
    for i in range(256):
        trans[i] = ceil(255*cum_list[i])
    return trans


def main():
    myPicture = 'test3.png'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im)
    pixels = pixels.transpose()

    # global equalization
    g_hist = global_hist(pixels)
    trans = trans_function(g_hist)
    print(trans)
    # plt.bar(list(range(256)),trans)
    # plt.show()
    im2 = Image.new('L', im.size)
    pixels2 = im2.load()
    for i in range(im.size[0]):
        for j in range(im.size[1]):
            pixels2[i,j] = int(trans[pixels[i,j]])
    im2.save('global equalization.jpg')
    im2.show()
    
    # local equalization
    for patch_size in (51,151,201):
        hist_list = local_hist(pixels,patch_size)
        print('local hist done!',patch_size)
        im3 = Image.new('L', im.size)
        pixels3 = im3.load()
        for row_index in range(hist_list.shape[0]):
            h = hist_list[row_index,:]
            i = int(h[0])
            j = int(h[1])
            # print(i,j)
            trans = trans_function(h[2:])
            pixels3[i,j] = int(trans[pixels[i,j]])
            # print(pixels[i,j],pixels3[i,j])
        im3.save('local equalization with patch size = '+str(patch_size)+'.jpg')
        # im3.show()
    

if __name__ == '__main__':
    main()