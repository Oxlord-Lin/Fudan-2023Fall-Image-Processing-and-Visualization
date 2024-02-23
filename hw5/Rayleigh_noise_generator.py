"""能够生成瑞利噪声，返回的噪声会进行归一化"""

import numpy as np

def generator_r(m:int, n:int, a:float = -1, b:float = 10) -> np.array:
    """
    generator of white noise
    :param: m is the height of your figure
    :param: n is the width of your figure
    :param: a is a constant add to the Rayleigh r.v., default -1
    :param: b is the scale of the pdf of Rayleigh distribution, default 10
    :return: an array of size m*n, containing real white noise, normalized to [-1,1]
    """
    noise = a + np.random.rayleigh(scale=b,size=(m,n))
    noise = noise/np.max(np.abs(noise))
    return noise
