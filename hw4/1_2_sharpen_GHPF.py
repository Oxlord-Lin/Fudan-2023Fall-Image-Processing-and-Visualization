import numpy as np
from numba import jit, njit  # 加速器
from PIL import Image
from math import exp


# @jit
def logRescale(M:np.array) -> np.array:
    """先取模得到频谱，然后取对数，并映射到[0,255]区间，将映射后的频谱转化为uint8后返回"""
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
    """映射到[0,255]区间，并转化为uint8格式返回"""
    # m,n = M.shape
    largest = np.max(M)
    least = np.min(M)
    M = 255 * (M - least)/(largest - least)
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M

# @jit
def sharpen(pixels: np.array, sigam:float = 10, const = 1) -> (np.array, np.array, np.array):
    """
    GHPF, low frequency are removed and high frequency remain, 
    using the Gauss frequency filter 
    :param: pixels are the indensity of all the pixels in your image
    :param: sigma is the standard deviation of your Gauss Kernel in frequency domain
    :param: const is a constant added to the Gauss filter
    :return: three arrays: the original frequency F, the filtered frequency H, and the filtered figure g
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

    # step 3: DFT and make Hadarmard product with Gauss filter
    F = np.fft.fft2(P)
    H = np.empty(F.shape) # Gause filter
    center = np.array([H.shape[0]//2, H.shape[1]//2])
    for x in range(H.shape[0]):
        for y in range(H.shape[1]):
            D2 = (x-center[0])**2 + (y-center[1])**2
            H[x,y] = 1 - exp(-D2/(2*sigam**2)) + const 
    G = F*H

    # step 4: IDFT
    g = np.fft.ifft2(G).real
    g = g*S
    g = g[0:m, 0:n]
    for x in range(g.shape[0]):
        for y in range(g.shape[1]): 
            # 截断法
            if g[x,y] > 255:
                g[x,y] = 255
            elif g[x,y] < 0:
                g[x,y] = 0
    
    # step 5: return the original frequency, the filtered frequency and the filtered figure
    return F, G, g


def main():
    myPicture = 'test.jpeg'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im) 
    sigmas = [1]
    const = 1

    for sigma in sigmas:
        # 锐化处理
        orig_freq, filter_freq, sharpen_pixels = sharpen(pixels,sigma,const)
        
        # 对频谱取模，取对数，映射到[0,255]区间，并转换为int8类型
        orig_freq = logRescale(orig_freq)
        filter_freq = logRescale(filter_freq)
        
        # 将滤波后的像素映射到[0,255]，并转换为uint8格式
        sharpen_pixels = rescale(sharpen_pixels)
        
        # 转成图像格式
        freq_im = Image.fromarray(orig_freq)
        filtered_freq_im = Image.fromarray(filter_freq)
        sharpen_im = Image.fromarray(sharpen_pixels)

        # 存储图像
        freq_im.save('sharpen_result_Gauss/'+'sigma='+str(sigma)+' c='+str(const)+' frequency'+'.png')
        filtered_freq_im.save('sharpen_result_Gauss/'+'sigma='+str(sigma)+' c='+str(const)+' filtered frequency'+'.png')
        sharpen_im.save('sharpen_result_Gauss/'+'sigma='+str(sigma)+' c='+str(const)+' filtered figure'+'.png')
    


if __name__ == '__main__':
    main()