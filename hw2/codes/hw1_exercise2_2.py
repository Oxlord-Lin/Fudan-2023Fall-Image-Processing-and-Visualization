"""implement computation of local histograms of an image 
using the efficient update of local histogram method introduced 
in local histogram processing.
Note that because only one row or column of the neighborhood changes 
in a one-pixel translation of the neighborhood, 
updating the histogram obtained in the previous location 
with the new data introduced at each motion step is 
possible and efficient in computation."""

from math import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



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

def draw_local_histogram(h,bins,w):
    """
    input:
        h: a one-dimension array, where the first 2 elements are the indexes,
            and the rest elements are the local histogram
        bins: the number of markerse on the axis
        w: the patch size
    output:
        no output. But this function can draw the local histogram 
            and show the indexes of the central pixel in the title
    """
    x = h[0]
    y = h[1]
    h = h[2:]
    plt.bar(list(range(bins)), h, color = "#4CAF50")
    t = 'the local histogram centered at (' + str(int(x))+ ',' + str(int(y)) + ')'\
        + ' with patch size=' + str(int(w))
    plt.title(t)
    # plt.show()
    plt.savefig(t+'.jpg')
    pass


def main():
    myPicture = 'test1.jpeg'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im)
    patch_size = 25
    bins = 256
    hist_list = local_hist(pixels,patch_size)
    # show the first 3, the middle 3, and the last 3 local histogram
    for i in [0, 1,2, pixels.size//2+20, pixels.size//2 + 21,pixels.size//2 + 22, -3,-2, -1]:
        h = hist_list[i,:]
        draw_local_histogram(h,bins,patch_size)


if __name__ == '__main__':
    main()