import numpy as np
# from numba import jit, njit  # 加速器
from PIL import Image
from math import pi


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
def LaplacianFilter(pixels: np.array) -> (np.array, np.array, np.array):
    """
    Laplacian sharpen filter in frequency
    :param: pixels are the indensity of all the pixels in your image
    :return: three arrays: the original frequency F, the filtered frequency H, and the Laplacian g
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

    # step 3: DFT and make Hadarmard product with Laplace filter
    F = np.fft.fft2(P)
    H = np.zeros(F.shape) # Laplace filter
    center = np.array([H.shape[0]//2, H.shape[1]//2])
    for x in range(H.shape[0]):
        for y in range(H.shape[1]):
            D2 = (x-center[0])**2 + (y-center[1])**2
            # H[x,y] = -4 * pi**2 * (x**2 + y**2)
            H[x,y] = -4 * pi**2 * D2
    G = F*H

    # step 4: IDFT
    g = np.fft.ifft2(G).real
    # g = g*S
    g = g[0:m, 0:n] # g携带了边缘信息
    # for x in range(g.shape[0]):
    #     for y in range(g.shape[1]):
    #         if g[x,y] > 255:
    #             g[x,y] = 255
    #         elif g[x,y] < 0:
    #             g[x,y] = 0

    # 把g标准化到[-1,1]区间
    # gMin = np.min(g)
    gMax = np.max(g)
    g = g/gMax

    # step 5: return the original frequency, the filtered frequency and the filtered figure
    return F, G, g


def main():
    myPicture = '肝脏.png'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    pixels = np.array(im) 

    # 锐化操作
    orig_freq, filter_freq, lap = LaplacianFilter(pixels)
    
    # 对频域结果取模，取对数，映射到[0,255]区间，并转换为int8类型
    orig_freq = logRescale(orig_freq)
    filter_freq = logRescale(filter_freq)

    # 频谱转成图像格式
    freq_im = Image.fromarray(orig_freq)
    filtered_freq_im = Image.fromarray(filter_freq)
    # 存储频域图像
    freq_im.save('sharpen_result_Laplacian/'+'frequency'+'.png')
    filtered_freq_im.save('sharpen_result_Laplacian/'+'filtered frequency'+'.png')
    

    # 存储Laplace的图像
    # lap_trans = rescale(lap)
    lap_trans = logRescale(lap)
    lap_im = Image.fromarray(lap_trans)
    lap_im.save('sharpen_result_Laplacian/'+'laplace.png')
    

    # 将Laplace量加到原图像上，并把原图像rescal到[0,255]区间上
    for c in [-5,-10,-15,-20,-50]:
        sharpen_pixels = pixels + c*lap
        
        # 截断法
        sharpen_pixels = np.uint8(sharpen_pixels)
        for i in range(sharpen_pixels.shape[0]):
            for j in range(sharpen_pixels.shape[1]):
                p = sharpen_pixels[i,j]
                if p > 255:
                    sharpen_pixels[i,j] = 255
                elif p < 0:
                    sharpen_pixels[i,j] = 0
        
        # 放缩法
        # sharpen_pixels = rescale(sharpen_pixels)
        
        sharpen_im = Image.fromarray(sharpen_pixels)
        # 存储图像
        sharpen_im.save('sharpen_result_Laplacian/'+'c='+str(c)+' sharpen figure.png')
    


if __name__ == '__main__':
    main()