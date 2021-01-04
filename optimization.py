import trimesh
import numpy as np

def fill_voxels(mesh, voxel):
    global relevant_voxels
    global shell
    global k
    shell = np.copy(voxel.matrix)
    mat = voxel.matrix
    for slice in range(mat.shape[0]):
        for row in range(mat.shape[1]):
            points = [[slice, row, cell] for cell in range(mat.shape[2])]
            points = voxel.indices_to_points(np.asarray(points))
            contains = mesh.contains(points)
            mat[slice][row][contains] = True
    relevant_voxels = voxel.matrix ^ shell
    k = np.sum(relevant_voxels)
    return trimesh.voxel.VoxelGrid(voxel.matrix)

def voxelize_and_fill(obj):
    global pitch
    voxel_obj = obj.voxelized(pitch)
    return fill_voxels(obj, voxel_obj)

def get_voxel_obj_mass_properties(voxel_obj):
    return voxel_obj.as_boxes().mass_properties


def get_s(mass_properties):
    """
    not good
    computes s vector from mass properties
    :param mass_properties:
    :return:
    """
    s1 = mass_properties["mass"]
    sx, sy, sz = mass_properties["center_mass"]/s1
    sxy = -mass_properties["inertia"][0,1]
    syz = -mass_properties["inertia"][2,1]
    sxz = -mass_properties["inertia"][2,0]
    sx2 = (-mass_properties["inertia"][0,0]+mass_properties["inertia"][1,1]+mass_properties["inertia"][2,2])/2
    sy2 = (mass_properties["inertia"][0,0]-mass_properties["inertia"][1,1]+mass_properties["inertia"][2,2])/2
    sz2 = (mass_properties["inertia"][0,0]+mass_properties["inertia"][1,1]-mass_properties["inertia"][2,2])/2
    return [s1, sx, sy, sz, sxy, syz, sxz, sx2, sy2, sz2]

def compute_s_shell():
    """
    compute s vectors for all shell (relevant voxels)
    :return: matrix (1,10) of s vector for each relevant voxel
    """
    x, y, z = shell.nonzero()
    s_shell = np.zeros(10)
    indices = list(zip(x, y, z))
    for index in indices:
        new_mat = np.zeros(shell.shape, dtype=bool)
        new_mat[index] = True
        s_shell += s_singlebox(trimesh.voxel.VoxelGrid(new_mat).as_boxes().mass_properties)
    return s_shell

def compute_s():
    """
    compute s vectors for all k (relevant voxels)
    :return: matrix (k,10) of s vector for each relevant voxel
    """
    global map_index_to_num
    new_dict = dict([(value, key) for key, value in map_index_to_num.items()])
    s_k = np.zeros((len(map_index_to_num.items()),10))
    for index in new_dict:
        new_mat = np.zeros(relevant_voxels.shape, dtype=bool)
        new_mat[new_dict[index]] = True
        s_k[index] = s_singlebox(trimesh.voxel.VoxelGrid(new_mat).as_boxes().mass_properties)
    return s_k

def s_456(low1, high1, low2, high2):
    return (high1**2/2 - low1**2/2 ) * (high2**2/2 - low2**2/2 )

def s_squared(low, high):
    return (high**3 - low**3)/3

def s_singlebox(mass_properties):
    s1 = mass_properties["mass"]
    sx, sy, sz = mass_properties["center_mass"]/s1
    x_low = sx - 0.5
    x_high = sx + 0.5
    y_low = sy-0.5
    y_high = sy+0.5
    z_low = sz-0.5
    z_high = sz + 0.5
    sxy = s_456(x_high, x_low, y_high, y_low)
    syz = s_456(y_high, y_low, z_high, z_low)
    sxz = s_456(x_high, x_low, z_high, z_low)
    sx2 = s_squared(x_low, x_high)
    sy2 = s_squared(y_low, y_high)
    sz2 = s_squared(z_low, z_high)
    return [s1, sx, sy, sz, sxy, syz, sxz, sx2, sy2, sz2]


def laplacian():
    global relevant_voxels, map_index_to_num
    x,y,z = relevant_voxels.nonzero()
    indices = list(zip(x,y,z))
    laplacian = np.zeros((len(x), len(x)))
    map_index_to_num = {key: value for (value, key) in enumerate(indices)}
    x_lim, y_lim, z_lim = relevant_voxels.shape
    for index in indices:
        x,y,z = index
        count = 0
        if( x < x_lim-1 and (x + 1, y, z) in indices):
            count +=1
            laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x + 1, y, z)]] = -1
        if ( x > 0 and (x - 1, y, z) in indices):
            count += 1
            laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x - 1, y, z)]] = -1
        if ( y < y_lim-1 and (x , y + 1, z) in indices):
            count += 1
            laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x, y + 1, z)]] = -1
        if ( y > 0 and (x, y-1, z) in indices):
            count += 1
            laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x, y - 1, z)]] = -1
        if ( z < z_lim -1 and (x, y, z+1) in indices):
            count += 1
            laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x, y, z + 1)]] = -1
        if ( z > 0 and (x, y, z - 1) in indices):
            count += 1
            laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x, y, z - 1)]] = -1
        laplacian[map_index_to_num[(x,y,z)]][map_index_to_num[(x,y,z)]] = count
    return laplacian




# def optimization(s):

if __name__ == '__main__':
    pitch = 0.04
    relevant_voxels = None
    map_index_to_num = None
    filename = 'ellipsoid2_3.obj'
    filename = 'dolphin.obj'
    filename = 'kitten1_4.obj'
    obj = trimesh.load_mesh(filename)
    voxel_obj = voxelize_and_fill(obj)
    mass_properties = get_voxel_obj_mass_properties(voxel_obj)
    center_mass = mass_properties["center_mass"]
    mass = mass_properties["mass"]
    inertia = mass_properties["inertia"]
    s = get_s(mass_properties)
    s_shell = compute_s_shell()
    laplacian = laplacian()
    k_matrix = compute_s()
    s_everything = s_shell + np.sum(k_matrix, axis=0)
    beta = np.ones(k)
    s_current = s_everything - np.dot(beta, k_matrix)

    print(s_shell)
    print(s_current)