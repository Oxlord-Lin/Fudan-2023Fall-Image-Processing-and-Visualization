# Exercise1: piecewise linear transformation
from PIL import Image

def piecewise_transf(intensity, r1, r2, s1, s2):
    if intensity < r1:
        newIntensity = s1*(intensity)/r1
    elif intensity <= r2:
        newIntensity = s1 + (s2-s1)*(intensity-r1)/(r2-r1)
    else: # intensity > r2
        newIntensity = s2 + (255-s2)*(intensity-r2)/(255-r2)
    return int(newIntensity)


def intensity_transformation(pictureName,r1,r2,s1,s2):
    im_raw = Image.open(pictureName)
    im = im_raw.convert('L')
    pixels = im.load()
    for i in range(im.size[0]): # for every pixel:
        for j in range(im.size[1]):
            pixels[i,j] = piecewise_transf(pixels[i,j],r1,r2,s1,s2)

    im.save('piecewise linear transformation for'+str(pictureName))
    im.show()


def main():
    myPicture = 'test1.jpeg'
    intensity_transformation(myPicture,5,40,5,80)


if __name__ == '__main__':
    main()