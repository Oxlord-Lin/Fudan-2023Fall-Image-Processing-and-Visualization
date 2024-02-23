"""实现了最佳陷波滤波器"""

import numpy as np 
from PIL import Image
from numba import njit # 加速器

# 辅助函数
def truncate(M:np.array) -> np.array:
    """将超出[0,255]区间的灰度值进行截断，并返回uint8类型"""
    M = np.clip(M,0,255) # 截断
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M

# 辅助函数
def rescale(M:np.array) -> np.array:
    """线性映射到[0,255]区间，并返回uint8类型"""
    # m,n = M.shape
    largest = np.max(M)
    least = np.min(M)
    M = 255 * (M - least)/(largest - least)
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M

# 辅助函数
def logRescale(M:np.array) -> np.array:
    """先取模得到频谱，然后取对数，并映射到[0,255]区间，将映射后的频谱转为uint8返回"""
    # m,n = M.shape
    M = np.abs(M) # 取模得到频谱
    M = np.log(M + 1) # 取对数
    largest = np.max(M)
    least = np.min(M)
    M = 255 * (M - least)/(largest - least)
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M

# 辅助函数
def get_freq_spectrum(pixels:np.array):
    """得到频域"""
    m,n = pixels.shape
    # padding
    P = np.zeros((2*m,2*n))
    P[0:m,0:n] = pixels
    # DFT and centerization
    F = np.fft.fft2(P)
    F = np.fft.fftshift(F)
    return F

@njit
def best_notch_filter(pixels,noise,patchSize:int):
    """最佳陷波滤波器，要求patchSize为奇数"""
    X,Y  = pixels.shape
    half = patchSize//2
    w = np.zeros((X,Y))
    for i in range(half,X-half): # 不考虑边缘区域
        for j in range(half,Y-half):
            g = pixels[i-half:i+half+1,j-half:j+half+1]
            n = noise[i-half:i+half+1,j-half:j+half+1]
            gn = g*n
            n2 = n*n
            n_mean = np.average(n)
            n2_mean = np.average(n2)
            gn_mean = np.average(gn)
            g_mean = np.average(g)
            w[i,j] = (gn_mean - g_mean*n_mean)/(n2_mean - n_mean + 0.001)
            # print(w[i,j])
            # time.sleep(0.05)
    # print(w)
    return pixels - w*noise
    # return pixels - noise


def main():
    im_raw = Image.open("带噪声的脑膜图.png")
    im = im_raw.convert('L')
    pixels = np.array(im)

    # 寻找噪声模式（展示带噪声图像的频谱）
    F = get_freq_spectrum(pixels)
    F_log = logRescale(F)
    im_ori_freq_spectrum = Image.fromarray(F_log)
    im_ori_freq_spectrum.save('notch_filter_result/'+'原图的频谱.png')


    # 构建针对噪声的频域滤波器（展示并保存频域滤波器）
    m,n = pixels.shape
    H = np.ones((2*m,2*n))
    H[m-90:m+90,:] = 0
    H[:,n-90:n+90] = 0

    H_rescale = rescale(H)
    im_H = Image.fromarray(H_rescale)
    im_H.save('notch_filter_result/'+'噪声滤波器示意图.png')


    # 展示噪声的干扰模式（噪声的频谱以及噪声的空间图像）
    noise_freq = F * H
    noise_freq_log = logRescale(noise_freq)
    im_noise_freq_log = Image.fromarray(noise_freq_log)
    im_noise_freq_log.save('notch_filter_result/'+'噪声的频域图像.png')

    noise_freq = np.fft.ifftshift(noise_freq) # decenterization
    noise = np.fft.ifft2(noise_freq).real # IDFT and remain the real part
    noise = noise[0:m,0:n] # remain the left top part
    noise_rescale = rescale(noise)
    im_noise = Image.fromarray(noise_rescale)
    im_noise.save('notch_filter_result/'+"噪声图像.png")


    # 使用最佳陷波滤波器进行空间过滤
    for patchSize in [3,5,7,9,11,13]:
        filtered_figure = best_notch_filter(pixels,noise,patchSize)

        # 保存滤波后的图像
        # filtered_figure = rescale(filtered_figure)
        filtered_figure = truncate(filtered_figure) 
        # filtered_figure = global_equalization(filtered_figure)
        im = Image.fromarray(filtered_figure)
        im.save('notch_filter_result/'+'滤波后的图像 patchSize='+str(patchSize) +'.png')


if __name__ == '__main__':
    main()