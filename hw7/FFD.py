"""实现了FFD形变算法"""

import numpy as np
from collections import defaultdict


def ffd(im_height:int,im_width:int, m:int, n:int, shift_dict:dict) -> np.array:
    """
    实现FFD形变算法，Y=X+Q(x)，其中Q(x)= \sum_i (shift_i * weight_i)
    input:
        im_height: 原图的高
        im_width: 原图的宽度
        m,n: 网格的高和宽，将会在原图上划分出(m+1)*(n+1)的均匀等分控制点
        shift_dict: 字典，用于传入所有发生位移的控制点，格式为{(i,j): delta_x, delta_y}，
            其中 (i,j) 表示第i行，第j列的控制点，
            delta_x, delta_y 分别表示在第一个维度和第二个维度上发生的位移
    output:
        shifted_index: 一个三维张量，大小为 im_height * im_width * 2，记录原图的每一个像素在FFD形变后的新坐标
    """

    ## ---------------------第一步，定义用于计算权重的B样条核函数---------------
    def beta(a,index):
        if index == -1: return (1-a)**3 / 6
        if index == 0:  return (3*a**3 - 6 * a**2 + 4)/6
        if index == 1:  return (-3*a**3 + 3*a**2 + 3*a +1)/6
        if index == 2:  return a**3/6
    

    ## -------------第二步，构建用于存储所有控制点位移的defaultdict--------------
    def return_no_shift():
        return np.zeros(2)
    
    # 使用默认字典存储所有发生位移的控制点，可以节省存储空间
    # 而且可以巧妙避免后面计算位移加权和时可能发生的越界问题
    shift = defaultdict(return_no_shift) 

    for item in shift_dict.items():
        i,j = item[0]
        delta_x,delta_y = item[1]
        shift[(i,j)] = np.array([delta_x,delta_y]) # 记录发生位移的控制点


    ## ---------------第三步，计算每个像素点在形变后的坐标-----------------------
    
    lx = im_height/m # 网格单元大小
    ly = im_width/n 

    # 用于记录每个【像素点】位移后的坐标

    # 初始化位移矩阵
    shifted_index = np.zeros((im_height,im_width,2))
    for x in range(im_height):
        for y in range(im_width):
            shifted_index[x,y,:] = np.array([x,y])  

    for x in range(im_height): # 遍历原图的每一个像素点
        dist_1 = (x-0)/lx
        ix = int(dist_1)
        u = dist_1 - ix # 距离最近的左侧控制点的距离除以lx

        for y in range(im_width):
            dist_2 = (y-0)/ly
            iy = int(dist_2)
            v = dist_2 - iy # 距离最近的上方控制点的距离除以ly

            for p in [-1,0,1,2]: # 对该像素点计算形变后的坐标
                for q in [-1,0,1,2]:
                    shifted_index[x,y,:] += shift[(ix+p,iy+q)] * beta(u,p) * beta(v,q) 
                    # 由于使用defaultdict，因此不会发生越界问题；越界的部分都统一返回[0,0]
    
    return shifted_index