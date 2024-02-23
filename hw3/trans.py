import numpy as np
from PIL import Image

def main():
    myPicture = 'OIP-C.jfif'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('L')
    im.save('test2.png')

if __name__ == '__main__':
    main()