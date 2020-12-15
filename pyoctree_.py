# Imports
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import sys, vtk
# sys.path.append('../')/
import pyoctree
from pyoctree import pyoctree as ot

print('pyoctree version = ', pyoctree.__version__)
print('vtk version = ', vtk.vtkVersion.GetVTKVersion())
# Read in stl file using vtk
reader = vtk.vtkSTLReader()
reader.SetFileName("Head.stl")
reader.MergingOn()
reader.Update()
stl = reader.GetOutput()
print("Number of points    = %d" % stl.GetNumberOfPoints())
print("Number of triangles = %d" % stl.GetNumberOfCells())

# Extract polygon info from stl

# 1. Get array of point coordinates
numPoints = stl.GetNumberOfPoints()
pointCoords = np.zeros((numPoints, 3), dtype=float)
for i in range(numPoints):
    pointCoords[i, :] = stl.GetPoint(i)

# 2. Get polygon connectivity
numPolys = stl.GetNumberOfCells()
connectivity = np.zeros((numPolys, 3), dtype=np.int32)
for i in range(numPolys):
    atri = stl.GetCell(i)
    ids = atri.GetPointIds()
    for j in range(3):
        connectivity[i, j] = ids.GetId(j)

tree = ot.PyOctree(pointCoords,connectivity)

print("Size of Octree               = %.3fmm" % tree.root.size)
print("Number of Octnodes in Octree = %d" % tree.getNumberOfNodes())
print("Number of polys in Octree    = %d" % tree.numPolys)

print(tree.root)
tree.root.branches
print(tree.root.branches[0])

tree.getOctreeRep()



file_name = "C:\\Users\\inbal\\PycharmProjects\\3D\\venv\\octree.vtu"
reader = vtk.vtkXMLUnstructuredGridReader()
# reader = vtk.vtkSphereSource()
reader.SetFileName(file_name)
reader.Update()
output = reader.GetOutput()
import vtk

'''

# create a rendering window and renderer
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)


# mapper
mapper = vtk.vtkOpenGLGlyph3DHelper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper.SetInput(reader.GetOutput())
else:
    mapper.SetInputConnection(reader.GetOutputPort())

# actor
actor1 = vtk.vtkActor()
actor1.SetMapper(mapper)
actor1.GetProperty().EdgeVisibilityOn()


# outline
outline = vtk.vtkOutlineFilter()
if vtk.VTK_MAJOR_VERSION <= 5:
    outline.SetInputData(reader.GetOutput())
else:
    outline.SetInputConnection(reader.GetOutputPort())
mapper2 = vtk.vtkPolyDataMapper()
if vtk.VTK_MAJOR_VERSION <= 5:
    mapper2.SetInput(outline.GetOutput())
else:
    mapper2.SetInputConnection(outline.GetOutputPort())

actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)
actor2.GetProperty().EdgeVisibilityOn()


# assign actor to the renderer
ren.AddActor(actor1)
ren.AddActor(actor2)

# enable user interface interactor
iren.Initialize()
renWin.Render()
iren.Start()
'''


mapper = vtk.vtkDataSetMapper()
mapper.SetInputConnection(reader.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
# actor.GetProperty().SetDiffuseColor(namedColors.GetColor3d("Tomato"))
# actor.GetProperty().SetEdgeColor(namedColors.GetColor3d("IvoryBlack"))
actor.GetProperty().EdgeVisibilityOn()
actor.GetProperty().SetOpacity(0.2)

    # sphereSource = vtk.vtkSphereSource()
    # sphereSource.SetRadius(0.02)

    # glyph3D = vtk.vtkGlyph3D()
    # glyph3D.SetInputData(uGrid)
    # glyph3D.SetSourceConnection(sphereSource.GetOutputPort())
    # glyph3D.ScalingOff()
    # glyph3D.Update()
    #
    # glyph3DMapper = vtk.vtkDataSetMapper()
    # glyph3DMapper.SetInputConnection(glyph3D.GetOutputPort())
    # glyph3DMapper.ScalarVisibilityOff()
    #
    # glyph3DActor = vtk.vtkActor()
    # glyph3DActor.SetMapper(glyph3DMapper)
    # glyph3DActor.GetProperty().SetColor(
    #     namedColors.GetColor3d("Banana"))

# textProperty = vtk.vtkTextProperty()
# textProperty.SetFontSize(24)

    # ss = "# of Tetras: " + str(numTets)
    # textMapper = vtk.vtkTextMapper()
    # textMapper.SetInput(ss)
    # textMapper.SetTextProperty(textProperty)

    # textActor = vtk.vtkActor2D()
    # textActor.SetMapper(textMapper)
    # textActor.SetPosition(10, 400)

    # Visualize
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.SetWindowName("Quadratic Tetra Demo")
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(640, 512)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

# widget = vtk.vtkSliderWidget()
# MakeWidget(widget, tessellate, textMapper, interactor)

renderer.AddActor(actor)
    # renderer.AddActor(glyph3DActor)
    # renderer.AddViewProp(textActor)
    # renderer.SetBackground(namedColors.GetColor3d("SlateGray"))

renderWindow.Render()
interactor.Start()

polys = vtk.vtkAppendPolyData()

Nslices = 10
p01 = (0,0,0)
p02 = (0,1,0)
for ks in range(Nslices):
    p0 = (0,p01[1]+(p02[1]-p01[1])/float(Nslices)*ks,0)
    plane = vtk.vtkPlane()
    plane.SetNormal(0,1,0)
    plane.SetOrigin(p0)

    cut = vtk.vtkCutter()
    cut.SetInput(reader)
    cut.SetCutFunction(plane)
    cut.Update()
    output = cut.GetOutput()
    polys.AddInput(output)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName('afile.vtp')
writer.SetInput(polys)
writer.Write()