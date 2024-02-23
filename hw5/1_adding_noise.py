import numpy as np 
from PIL import Image
from white_noise_generator import generator_w
from Rayleigh_noise_generator import generator_r


def truncate(M:np.array) -> np.array:
    """将超出[0,255]区间的灰度值进行截断，并返回uint8类型"""
    M = np.clip(M,0,255) # 截断
    M = np.uint8(M) # 无符号整数可以表示0~255
    return M


def main():
    for fileName in ['大脑图像.png','心脏图像.png']:
        im_raw = Image.open(fileName)
        im = im_raw.convert('L')
        pixels = np.array(im)
        m = pixels.shape[0]
        n = pixels.shape[1]
        for c in [50,100,150]:
            pixels_w_u_noise = pixels + c*generator_w(m,n,rv_type='uniform')
            im1 = Image.fromarray(truncate(pixels_w_u_noise))
            im1.save('adding_noise_result/'+fileName[0:4]+' c='+str(c)+' uniform white noise.png')
            pixels_w_g_noise = pixels + c*generator_w(m,n,rv_type='Gauss')
            im2 = Image.fromarray(truncate(pixels_w_g_noise))
            im2.save('adding_noise_result/'+fileName[0:4]+' c='+str(c)+' Gauss white noise.png')
            pixels_r_noise = pixels + c*generator_r(m,n)
            im3 = Image.fromarray(truncate(pixels_r_noise))
            im3.save('adding_noise_result/'+fileName[0:4]+' c='+str(c)+' Rayleigh noise.png')


if __name__ == '__main__':
    main()