import numpy as np
# from numba import jit, njit  # 加速器
from PIL import Image
from math import exp


# @jit
def logRescale(M:np.array) -> np.array:
    """先取模得到频谱，然后取对数，并映射到[0,255]区间，将映射后的频谱返回"""
    # m,n = M.shape
    M = np.abs(M) # 取模得到频谱
    M = np.log(M + 1) # 取对数
    largest = np.max(M)
    least = np.min(M)
    M = 255 * (M - least)/(largest - least)
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M

# @jit
def rescale(M:np.array) -> np.array:
    """映射到[0,255]区间"""
    # m,n = M.shape
    largest = np.max(M)
    least = np.min(M)
    M = 255 * (M - least)/(largest - least)
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M

# @jit
def removeNoise(pixels: np.array) -> (np.array, np.array, np.array):
    """
    remove noise from the target noisy picture provided by Prof. Zhuang
    :param: pixels are the indensity of all the pixels in the noise image
    :return: three arrays: the original frequency F, the filtered frequency G, and the filtered figure
    """
    # step 1: padding, double size
    m,n = pixels.shape
    P = np.zeros((2*m, 2*n))
    P[0:m, 0:n] = pixels # put the original figure on the left-top

    # step 2: preparation for shift
    S = np.ones((2*m, 2*n))
    for x in range(1,S.shape[0],2):
        S[x,:] *= -1
    for y in range(1,S.shape[1],2):
        S[:,y] *= -1
    P = P * S

    # step 3: DFT and make Hadarmard product with a special filter
    F = np.fft.fft2(P)
    # print(F.shape)

    H = np.zeros(F.shape) 
    # F_log = logRescale(F)
    H = np.empty(F.shape) # Gause filter
    center = np.array([H.shape[0]//2, H.shape[1]//2])
    sigma = 28
    for x in range(H.shape[0]):
        for y in range(H.shape[1]):
            H[x,y] = exp(-(x-center[0])**2/(2*sigma**2)) + exp(-(y-center[1])**2/(2*sigma**2))


    G = F*H

    # step 4: IDFT
    g = np.fft.ifft2(G).real
    # g = g*S
    g = g[0:m, 0:n]
    for x in range(g.shape[0]):
        for y in range(g.shape[1]):
            if g[x,y] > 255:
                g[x,y] = 255
            elif g[x,y] < 0:
                g[x,y] = 0
    # step 5: return the original frequency, the filtered frequency and the filtered figure
    return F, G, g


def main():
    myPicture = '作业四 图像.PNG'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im) 
    # print(pixels.shape)

    # 去噪
    orig_freq, filter_freq, p = removeNoise(pixels)
    
    # 对频域结果取模，取对数，映射到[0,255]区间，并转换为int8类型
    orig_freq = logRescale(orig_freq)
    filter_freq = logRescale(filter_freq)

    # 频谱转成图像格式
    freq_im = Image.fromarray(orig_freq)
    filtered_freq_im = Image.fromarray(filter_freq)
    # 存储频域图像
    freq_im.save('remove_noise_result/'+'frequency'+'.png')
    filtered_freq_im.save('remove_noise_result/'+'filtered frequency'+'.png')    

    # 存储去噪后的图像
    filtered_pixels = rescale(p)
    filtered_im = Image.fromarray(filtered_pixels)
    # 存储图像
    filtered_im.save('remove_noise_result/'+'result.png')
    

if __name__ == '__main__':
    main()