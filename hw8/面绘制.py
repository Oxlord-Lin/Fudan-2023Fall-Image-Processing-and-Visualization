"""面绘制"""

import nibabel as nib 
import vtk


img1 = nib.load('image_lr.nii') # load and save
img1_data = img1.get_fdata() # 获取标量场数据
dims = img1.shape #[124,124,73] # 数据场的维度
spacing = (img1.header['pixdim'][1],img1.header['pixdim'][2],img1.header['pixdim'][3]) # 间隔

image = vtk.vtkImageData() # 生成vtkImageDate对象
image.SetDimensions(dims[0],dims[1],dims[2]) # 设置vtkImageData对象的维度
image.SetSpacing(spacing[0],spacing[1],spacing[2]) # 设置间隔
image.SetOrigin(0,0,0)

if vtk.VTK_MAJOR_VERSION <= 5:
    image.SetNumberOfScalarComponents(1) # vtkImageData schalarArray tuple size
    image.SetScalarTypeToDouble()
else:
    image.AllocateScalars(vtk.VTK_DOUBLE,1)

# fill every entry of the image data
for z in range(dims[2]):
    for y in range(dims[1]):
        for x in range(dims[0]):
            # 将图像标量场数据填入vtkImageData对象的scalar属性中
            scalarData = img1_data[x][y][z]
            image.SetScalarComponentFromDouble(x,y,z,0,scalarData) 

Extractor = vtk.vtkMarchingCubes() # 移动立方体算法对象，得到等值面
Extractor.SetInputData(image) # 输入数据
Extractor.SetValue(0,150) # 设置value，求value=150的等值面

stripper = vtk.vtkStripper() # 建立三角带对象
stripper.SetInputConnection(Extractor.GetOutputPort()) # 输入数据，将生成的三角片连接成三角带

mapper = vtk.vtkPolyDataMapper() 
mapper.SetInputConnection(stripper.GetOutputPort())
# mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)

actor.GetProperty().SetColor(1,1,0)
actor.GetProperty().SetOpacity(0.95)
actor.GetProperty().SetAmbient(0.05)
actor.GetProperty().SetDiffuse(0.5)
actor.GetProperty().SetSpecular(1.0)

ren = vtk.vtkRenderer()
ren.SetBackground(1,1,1)
ren.AddActor(actor)
renWin = vtk.vtkRenderWindow()

renWin.AddRenderer(ren)
renWin.SetSize(750,750)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Initialize()
renWin.Render()
iren.Start()
