import trimesh
import numpy as np

def fill_voxels(mesh, voxel):
    global relevant_voxels
    mat = voxel.matrix
    for slice in range(mat.shape[0]):
        for row in range(mat.shape[1]):
            points = [[slice, row, cell] for cell in range(mat.shape[2])]
            points = voxel.indices_to_points(np.asarray(points))
            contains = mesh.contains(points)
            mat[slice][row][contains] = True
    relevant_voxels = voxel.matrix
    return trimesh.voxel.VoxelGrid(voxel.matrix)

def voxelize_and_fill(obj):
    global pitch
    voxel_obj = obj.voxelized(pitch)
    return fill_voxels(obj, voxel_obj)

def get_voxel_obj_mass_properties(voxel_obj):
    return voxel_obj.as_boxes().mass_properties

def get_s(mass_properties):
    s1 = mass_properties["mass"]
    sx, sy, sz = mass_properties["center_mass"]/s1
    sxy = -mass_properties["inertia"][0,1]
    syz = -mass_properties["inertia"][2,1]
    sxz = -mass_properties["inertia"][2,0]
    sx2 = (-mass_properties["inertia"][0,0]+mass_properties["inertia"][1,1]+mass_properties["inertia"][2,2])/2
    sy2 = (mass_properties["inertia"][0,0]-mass_properties["inertia"][1,1]+mass_properties["inertia"][2,2])/2
    sz2 = (mass_properties["inertia"][0,0]+mass_properties["inertia"][1,1]-mass_properties["inertia"][2,2])/2
    return [s1, sx,sy,sz,sxy,syz,sxz, sx2, sy2, sz2]

# def optimization(s):

def main(filename):
    obj = trimesh.load_mesh(filename)
    voxel_obj = voxelize_and_fill(obj)
    mass_properties = get_voxel_obj_mass_properties(voxel_obj)
    center_mass = mass_properties["center_mass"]
    mass = mass_properties["mass"]
    inertia = mass_properties["inertia"]
    s = get_s(mass_properties)



if __name__ == '__main__':
    pitch = 0.04
    relevant_voxels = None
    filename = 'ellipsoid2_3.obj'
    #filename = 'dolphin.obj'
    main(filename)