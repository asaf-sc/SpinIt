import os

import trimesh
import numpy as np
import binvox_rw
import slqp


def fill_voxels(mesh, voxel):
    """
    fill the voxels that are inside the mesh
    :param mesh:
    :param voxel:
    :return:
    """
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
    return trimesh.voxel.VoxelGrid(voxel.matrix)


def voxelize_and_fill(obj):
    pitch = 0.04
    voxel_obj = obj.voxelized(pitch)
    return fill_voxels(obj, voxel_obj)


def get_voxel_obj_mass_properties(voxel_obj):
    return voxel_obj.as_boxes().mass_properties


def compute_s_shell():
    """
    compute s vectors for all shell (relevant voxels)
    :return: matrix (1,10) of s vector for each relevant voxel
    """
    x, z, y = shell.matrix.nonzero()
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
    s_k = np.zeros((len(map_index_to_num.items()), 10))
    for index in new_dict:
        new_mat = np.zeros(relevant_voxels.shape, dtype=bool)
        new_mat[new_dict[index]] = True
        s_k[index] = s_singlebox(trimesh.voxel.VoxelGrid(new_mat).as_boxes().mass_properties)
    return s_k


def s_two_variables(low1, high1, low2, high2):
    return (high1 ** 2 / 2 - low1 ** 2 / 2) * (high2 ** 2 / 2 - low2 ** 2 / 2)


def s_variable_squared(low, high):
    return (high ** 3 - low ** 3) / 3


def s_singlebox(mass_properties):
    """
    compute 10-vector integrals for a signle voxel
    :param mass_properties:
    :return:
    """
    global p
    s1 = mass_properties["mass"]
    sx, sz, sy = mass_properties["center_mass"] / s1 - p
    x_low = sx - 0.5
    x_high = sx + 0.5
    y_low = sy - 0.5
    y_high = sy + 0.5
    z_low = sz - 0.5
    z_high = sz + 0.5
    sxy = s_two_variables(x_high, x_low, y_high, y_low)
    syz = s_two_variables(y_high, y_low, z_high, z_low)
    sxz = s_two_variables(x_high, x_low, z_high, z_low)
    sx2 = s_variable_squared(x_low, x_high)
    sy2 = s_variable_squared(y_low, y_high)
    sz2 = s_variable_squared(z_low, z_high)
    return [s1, sx, sy, sz, sxy, syz, sxz, sx2, sy2, sz2]


def s_singlebox_shifted(mass_properties):
    global p
    s1 = mass_properties["mass"]
    sx, sz, sy = mass_properties["center_mass"] / s1
    Icom = mass_properties["inertia"]
    I = Icom + ((sz * sz) / s1) * np.diag([1, 1, 0])
    sxy = -I[0][1]
    syz = -I[1][2]
    sxz = -I[1][2]
    sx2 = (-I[0][0] + I[1][1] + I[2][2]) / 2
    sy2 = (I[0][0] - I[1][1] + I[2][2]) / 2
    sz2 = (I[0][0] + I[1][1] - I[2][2]) / 2
    return [s1, sx, sy, sz, sxy, syz, sxz, sx2, sy2, sz2]


def laplacian():
    """
    Compute laplacian of neighboring cells
    :return:
    """
    global relevant_voxels, map_index_to_num
    x_, z_, y_ = relevant_voxels.nonzero()
    indices = list(zip(x_, y_, z_))
    laplacian = np.zeros((len(x_), len(x_)))
    map_index_to_num = {key: value for (value, key) in enumerate(indices)}
    x_lim, z_lim, y_lim = relevant_voxels.shape
    for ind in indices:
        x_, y_, z_ = ind
        count = 0
        if x_ < x_lim - 1 and (x_ + 1, y_, z_) in indices:
            count += 1
            laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_ + 1, y_, z_)]] = -1
        if x_ > 0 and (x_ - 1, y_, z_) in indices:
            count += 1
            laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_ - 1, y_, z_)]] = -1
        if y_ < y_lim - 1 and (x_, y_ + 1, z_) in indices:
            count += 1
            laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_, y_ + 1, z_)]] = -1
        if y_ > 0 and (x_, y_ - 1, z_) in indices:
            count += 1
            laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_, y_ - 1, z_)]] = -1
        if z_ < z_lim - 1 and (x_, y_, z_ + 1) in indices:
            count += 1
            laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_, y_, z_ + 1)]] = -1
        if z_ > 0 and (x_, y_, z_ - 1) in indices:
            count += 1
            laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_, y_, z_ - 1)]] = -1
        laplacian[map_index_to_num[(x_, y_, z_)]][map_index_to_num[(x_, y_, z_)]] = count
    return laplacian


def get_origin(shell):
    """
    find contant point of the top
    calculated as the lowest point on axis z
    :param shell:
    :return:
    """
    matrix = shell.matrix[:, 0, :]
    return matrix.nonzero()[0][0] + 1, 0, matrix.nonzero()[1][0] + 1


if __name__ == '__main__':
    size = 20
    relevant_voxels = None
    map_index_to_num = None

    filename = 'ellipsoid'
    # filename = 'dolphin'
    # filename = 'kitten'

    if os.path.exists(f"{filename}.binvox"):
        os.remove(f"{filename}.binvox")
    os.system(f"binvox.exe -d {size} {filename}.obj")

    with open(f"{filename}.binvox", 'rb') as f:
        binvox_obj = binvox_rw.read_as_3d_array(f)

    voxel_obj = trimesh.voxel.VoxelGrid(binvox_obj.data)
    shell = trimesh.voxel.VoxelGrid(binvox_obj.data).hollow()
    relevant_voxels = voxel_obj.matrix ^ shell.matrix
    p = get_origin(shell)

    s_shell = compute_s_shell()
    laplacian = laplacian()
    s_omega_k_matrix = compute_s()
    s_everything = s_shell + np.sum(s_omega_k_matrix, axis=0)

    # run optimization solver
    beta_k = slqp.slqp(laplacian, s_everything, s_omega_k_matrix)

    # create new model from the optimization result
    map_index_to_num_inv = dict([(value, key) for key, value in map_index_to_num.items()])
    result_mat = voxel_obj.matrix.copy()
    for index in np.round(beta_k).nonzero()[0]:
        x, y, z = map_index_to_num_inv[index]
        result_mat[x, z, y] = False
    result = trimesh.voxel.VoxelGrid(result_mat)

    # show results
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:, :, :size / 4])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:, :, size / 4:])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:size / 4, :, :])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[size / 4:, :, :])).show()
    trimesh.voxel.VoxelGrid(np.asarray([result_mat[:, :, size / 4]])).show()  # display slice
    print(f"Initial mass properties for solid model:\n{voxel_obj.as_boxes().mass_properties}")
    print(f"Resulting model mass properties:\n{result.as_boxes().mass_properties}")
