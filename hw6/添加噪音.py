import numpy as np
from PIL import Image
import random

myPicture = '肝脏.png'
im_raw = Image.open(myPicture)
im = im_raw.convert('L')
pixels = np.array(im)
pixels = np.uint8(pixels)
for i in range(pixels.shape[0]):
    for j in range(pixels.shape[1]):
        u = random.uniform(0,1)
        if u<0.005:
            pixels[i,j] = 0
        if u>0.995:
            pixels[i,j] = 255
im = Image.fromarray(pixels)
im.save('受到椒盐噪声污染的肝脏图像.png')