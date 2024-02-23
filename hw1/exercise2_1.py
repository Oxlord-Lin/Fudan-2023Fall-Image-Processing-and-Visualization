"""implement n-dimensional joint histogram 
and test the code on two-dimensional data"""

from math import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def n_dimension_joint_histogram(data,Min,bins,Max):
    """
    input:
    data is a n*k array, where n is the dimension and k is the number of elements.
    Each element is n-dimension.
    bins is a list or a tuple which contains the number of blocks in each direction.
    Min contains the lower bound in each dimension.
    Max contains the upper bound in each dimension.
    output:
    hist: an n-dimension array of frequency
    edges: a list of length n, containing the markers of axises in n directions.
    """
    n, *temp = data.shape
    edges = []
    spacing = []
    for i in range(n):
        spacing.append( round((Max[i] - Min[i])/bins[i])  )
        edges.append(list(range(Min[i],Max[i]+spacing[i],spacing[i])))
    hist = np.zeros(bins) # 创建n维数组用于存放histogram
    for i in range(data.shape[i]): # 遍历所有element
        pos = []  # 用于确认该点的数据应处于联合直方图的哪个位置
        for k in range(n): # pos记录该点每个维度上的分量
            pos.append(int(data[k,i]//spacing[k]))
        hist[tuple(pos)] += 1
    return hist, edges


def main():
    myPicture = 'test2.jpg'
    im_raw = Image.open(myPicture)
    im = im_raw.convert('RGB')
    r,g,b = im.split()
    
    r = np.array(r)
    rv = r.ravel()
    g = np.array(g)
    gv = g.ravel()

    hist, edges = n_dimension_joint_histogram(np.vstack((rv,gv)),[0,0],[16,16],[255,255])
    xedges = np.array(edges[0])
    yedges = np.array(edges[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Construct arrays for the anchor positions of the bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 4, yedges[:-1] + 4, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the bars.
    dx = dy = 8 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_title('joint 3D histogram of my 2-dimension test data')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    plt.savefig('test2_3d_bar')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    H = ax.hist2d(rv,gv,bins=16)
    fig.colorbar(H[3],ax=ax)
    ax.set_title('joint histogram of my 2-dimension test data')
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    # plt.show()
    plt.savefig('test2_2d_colorful_histogram')


if __name__ == '__main__':
    main()