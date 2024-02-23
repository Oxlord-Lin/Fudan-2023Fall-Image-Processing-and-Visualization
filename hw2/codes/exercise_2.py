from math import *
import numpy as np
from PIL import Image
from hw1_exercise2_2 import local_hist # 从第一次的作业中导入高效计算Local histogram的函数

def local_adaptive_thresholding(pixel,hist):
    """
    功能：使用OTSU，基于局部histogram，判断pixel应该变成0还是255
    input: pixel是当前像素的灰度值，hist是局部直方图
    output: 0或者255
    """
    S = int(sum(hist)) 
    one_hist = hist/S # 归一化
    mu_global =  np.inner(one_hist, np.array(range(256)))
    T = 127 # 为了防止出现纯色的图片（比如灰度全都为255）而找不出T，这里先预设一个，防止报错
    sigma = -float('inf')
    flg = 0
    for t in range(1,255):
        if int(hist[t]) == 0:
            if flg == 1:
                continue  # 连续为0的区间只需要计算第一个作为阈值的情况，节省运算量
            else: # flg == 0
                flg == 1
        else: # int(hist[t]) != 0:
            flg == 0

        omega_bk = int(sum(hist[0:t])) 
        if omega_bk == 0:
            continue 
            # 当累积概率为0时，可以跳过
            # 为了避免浮点运算，则使用整数运算判断
        if omega_bk == S: # 当累积概率为1时，肯定不是分界值，可以直接结束迭代，节省时间
            if pixel > T:
                return 255
            else:
                return 0

        # 以下是OTSU算法的核心部分，该算法具有高效的更新方式
        omega_bk = sum(one_hist[0:t]) # 累积概率
        mean_bk = np.inner(one_hist[0:t], np.array(range(t))) # accumulated mean
        sigma_new = (mu_global*omega_bk-mean_bk)**2 / (omega_bk*(1-omega_bk)) # 前景与背景的方差
        if sigma_new>sigma:
            sigma = sigma_new
            T = t
    if pixel > T:
        return 255
    else:
        return 0


def main():
    myPicture = '肝脏.png'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im) # 记得转置
    pixels = pixels.transpose()
    pixels2 = im.load() 

    # 【重要说明】
    # 在此处，我提供两种生成局部直方图的方式（默认使用第二种！）
    # 第一种方式直接调用第一次作业的代码，高效地求出当前图片的灰度值的所有局部直方图
    # 如需使用第一种方式，请将以下两行代码解除注释，然后将第二种方法注释掉。
    # patch_size = int(input('请输入patch size：'))
    # hist_list = local_hist(pixels, patch_size)

    # 第二种方式，则导入我已经提前准备好的不同patchSize对应的局部直方图文件
    # 只需要设置patch_size参数即可，更加方便
    # patch_size可以选择: 25, 50, 100, 150
    patch_size = int(input('请输入patch size（可选25，50，100，150）：')) 
    hist_list = np.load('some local histograms/'+myPicture+'hist_list_patchSize='+str(patch_size)+'.npy')


    # 以下为局部自适应二值化过程，需要比较久的时间，需耐心等待
    for i in range(hist_list.shape[0]//2):
        row = hist_list[2*i,:]
        row_parl = hist_list[2*i+1,:] # 引入并行的方法，提高运算速度，不然会等得更久
        x,y = int(row[0]), int(row[1])
        x_parl, y_parl = int(row_parl[0]), int(row_parl[1]) # 一次循环计算两个像素，利用空间局部性
        # print(x,y)
        # print(x_parl,y_parl)
        pixels2[x,y] = local_adaptive_thresholding(pixels[x,y],row[2:])
        pixels2[x_parl,y_parl] = local_adaptive_thresholding(pixels[x_parl,y_parl],row_parl[2:])
    
    if hist_list.shape[0]%2 == 1: # 对hist的最后一个元素进行二值化
        # print('像素数量为奇数！')
        row = hist_list[-1,:]
        x,y = int(row[0]), int(row[1])
        # print(x,y)
        pixels2[x,y] = local_adaptive_thresholding(pixels[x,y],row[2:])


    # 解除下面这行代码的注释后，会将二值化后的灰度值矩阵，自动命名并存储到本地
    # 以后就可以直接导入灰度值矩阵生成二值化后的图像，不用等那么久了
    # np.save('pixels after thresholding/'+'local_Thresholding_patchSize='+str(patch_size),np.array(im).transpose())

    im.save('exercise2_'+myPicture+'_patchSize='+str(patch_size)+'.jpg')



if __name__ == '__main__':
    main()