"""体绘制"""

import nibabel as nib 
import vtk
import numpy as np


img1 = nib.load('image_lr.nii') # load and save
img1_data = img1.get_fdata() # 获取标量场数据
dims = img1.shape #[124,124,73] # 数据场的维度
spacing = (img1.header['pixdim'][1],img1.header['pixdim'][2],img1.header['pixdim'][3]) # 间隔

image = vtk.vtkImageData() # 生成vtkImageDate对象
image.SetDimensions(dims[0],dims[1],dims[2]) # 设置vtkImageData对象的维度
image.SetSpacing(spacing[0],spacing[1],spacing[2]) # 设置间隔
image.SetOrigin(0,0,0)
image.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)

image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

intRange = (20,500) # 设置感兴趣的灰度区域
max_u_short = 128
const = max_u_short / np.float64(intRange[1]-intRange[0])
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            scalarData = img1_data[x][y][z]
            scalarData = np.clip(scalarData,intRange[0],intRange[1]) # 超出范围的部分进行截断
            scalarData = const * np.float64(scalarData - intRange[0])
            image.SetScalarComponentFromFloat(x,y,z,0,scalarData)

## 设置传输函数的参数

# create transfer mapping scalar value to opacity
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(0, 0.0)
opacityTransferFunction.AddSegment(23, 0.3, 128, 0.5)
opacityTransferFunction.ClampingOff()

# create transfer mapping scalar value to color
colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBSegment(0, 0.0, 0.0, 0.0, 20, 0.2, 0.2, 0.2)
colorTransferFunction.AddRGBSegment(20, 0.1, 0.1, 0, 128, 1, 1, 0) # intensity between 0

# grad to opacity transfer
gradientTransferFunction = vtk.vtkPiecewiseFunction()
gradientTransferFunction.AddPoint(0, 0.0)
gradientTransferFunction.AddSegment(100, 0.1, 1000, 0.3)


# the property dexcribes how the data will work
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetGradientOpacity(gradientTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear() # 采样时使用线性插值，性价比高
volumeProperty.SetAmbient(1)
volumeProperty.SetDiffuse(0.9) # 漫反射
volumeProperty.SetSpecular(0.8) # 镜面反射
volumeProperty.SetSpecularPower(10) # 用于描述镜面反射的强度

# The mapper / ray cast function know how to render the data.
volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
volumeMapper.SetInputData(image)
volumeMapper.SetImageSampleDistance(5.0)

# the volume holds the mapper and the property and can be used to position/orient the volume
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)

ren = vtk.vtkRenderer()
ren.SetBackground(1,1,1)
ren.AddVolume(volume)
renWin = vtk.vtkRenderWindow()

light = vtk.vtkLight()
light.SetColor(0,1,1)
ren.AddLight(light)

renWin.AddRenderer(ren)
renWin.SetSize(750,750)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

renWin.Render()
iren.Initialize()
iren.Start()