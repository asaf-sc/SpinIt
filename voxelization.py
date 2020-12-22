import trimesh
import numpy as np

filename = 'ellipsoid2_3.obj'
filename = 'dolphin.obj'
obj = trimesh.load_mesh(filename)

#TODO: fill mesh inside
#retruns new voxel object
def fill_voxels(points, mesh, voxel):
    mat = voxel.matrix
    for slice in range(mat.shape[0]):
        for row in range(mat.shape[1]):
            for cell in range(mat.shape[2]):
                if mat[slice][row][cell]==False:
                    # TODO: more efficient
                    point = voxel.indices_to_points(np.asarray([slice, row, cell]))
                    try:
                        mesh.contains([point])
                    except:
                        voxel.matrix[slice][row][cell] = True
                    # if mesh.contains([point]):
                    #     voxel.matrix[slice][row][cell] = True
    return trimesh.voxel.VoxelGrid(voxel.matrix)


p1 = np.asarray([0,0,0])
p2 = np.asarray([0,0,1])
a = p2-p1
M = obj.mass
c = obj.center_mass
g = 10
d = np.sqrt(c[0]**2+c[1]**2) #distance from input rotation axis
external_torque = M*g*d

inertia_tensor = obj.moment_inertia
components, vectors = trimesh.inertia.principal_axis(inertia_tensor)


obj.visual.face_colors = [100,100,100,100]
voxel_obj = obj.voxelized(0.01)

voxel_obj_filled = fill_voxels(None, obj, voxel_obj)
voxel_obj_filled.show()

axis = trimesh.creation.axis(origin_color=[1.,0,0])
scene = trimesh.Scene()
scene.add_geometry(obj)
scene.add_geometry(axis)
scene.show()

obj.apply_transform(trimesh.transformations.scale_matrix(2))
obj.show()
# viewer = trimesh.viewer.SceneViewer(scene)
# viewer.add_geometry(obj)
# viewer.toggle_axis()
#voxel_obj.show()

#to spin stably
#1. The center of mass c must lie on axis a
def cond1(c, p1, p2):
    # c is center of mass
    # a is spin axis
    res = np.cross(p2-p1, c-p1)/np.linalg.norm(p2-p1)
    return res == 0

def cond2():
    pass

def cond3(a , maximum_principa_inertia):
    # TODO: change to close to 0
    return np.cross(a, maximum_principa_inertia) == [0,0,0]

def cond4():
    pass


def f_yoyo (gamma_I, Ia, Ib, Ic):
    return gamma_I*(pow((Ia/Ic),2)+ pow((Ib/Ic),2))

def f_top (f_yoyo, gamma_c, l, M):
    return gamma_c*pow(l*M, 2)+f_yoyo