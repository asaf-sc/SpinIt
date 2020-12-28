import trimesh
import numpy as np


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

def get_mass(voxel):
    return np.sum(voxel.matrix)

def get_center_mass(voxel):
    mat = voxel.matrix
    cm = np.zeros(3)
    points = voxel.points
    cm = np.sum(points, axis =0)
    return cm/voxel.filled_count


def get_inertia_tensor(voxel, mass =1):
    mat = voxel.matrix
    I = np.zeros((3,3))
    points = voxel.points
    for x,y,z in points:
        r = np.asarray([[0, -z, y],
                        [z, 0, -x],
                         [-y, x, 0]])
        I -= np.matmul(r,r)*mass
    return I


#retruns new voxel object
def fill_voxels(points, mesh, voxel):
    mat = voxel.matrix
    for slice in range(mat.shape[0]):
        for row in range(mat.shape[1]):
            points = [[slice, row, cell] for cell in range(mat.shape[2])]
            points = voxel.indices_to_points(np.asarray(points))
            contains = mesh.contains(points)
            # print(contains)
            mat[slice][row][contains] = True
                    # if mesh.contains([point]):
                    #     voxel.matrix[slice][row][cell] = True
    relevant_voxels = voxel.matrix
    return trimesh.voxel.VoxelGrid(voxel.matrix)



filename = 'ellipsoid2_3.obj'
filename = 'dolphin.obj'
obj = trimesh.load_mesh(filename)
relevant_voxels = None
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

obj.visual.face_colors = [100,100,100,100] #make transparent
voxel_obj = obj.voxelized(0.01)

print("mesh:")
print(obj.mass_properties)

print("empty:")
print(voxel_obj.as_boxes().mass_properties)
calc_inertia = get_inertia_tensor(voxel_obj)


voxel_obj_filled = fill_voxels(None, obj, voxel_obj)
#voxel_obj_filled.show()
print("filled:")
print(voxel_obj_filled.as_boxes().mass_properties)

axis = trimesh.creation.axis(origin_color=[1.,0,0])
scene = trimesh.Scene()
# scene.add_geometry(voxel_obj_filled)
scene.add_geometry(axis)
scene.show()

# viewer = trimesh.viewer.SceneViewer(scene)
# viewer.add_geometry(voxel_obj_filled)
# viewer.toggle_axis()
voxel_obj.show()

