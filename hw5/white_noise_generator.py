"""能够生成白噪声：频谱为均值，相位角为随机变量，返回的噪声会进行归一化"""

import numpy as np
from math import pi

def generator_w(m:int,n:int, rv_type:str='uniform') -> np.array:
    """
    generator of white noise
    :param: m is the height of your figure
    :param: n is the width of your figure
    :param: const is the constant of the spectrum, default 1
    :param: rv_type is the type of R.V. of phase in the spectrum, it is allowed to be
    'uniform' or 'Gauss'(with miu = 0 and sigma = pi), default 'uniform'
    :return: an array of size m*n, containing real white noise, normalized to [-1,1]
    """
    const = 1
    noise_spectrum = const * np.ones((m,n))
    if rv_type == 'uniform':
        noise_phase = 1j * np.random.uniform(0,2*pi,(m,n))
        noise_phase = np.exp(noise_phase)
    elif rv_type == 'Gauss':
        noise_phase = 1j * np.random.normal(0,pi,(m,n))
        noise_phase = np.exp(noise_phase)
    
    # IDFT，并且只取实部
    noise = np.fft.ifft2(noise_spectrum*noise_phase).real
    # 归一化到 [-1,1] 区间
    noise = noise/np.max(np.abs(noise))
    return noise
    
