
import numpy as np
import binvox_rw
import trimesh

with open('kitten1_4_2.binvox', 'rb') as f:
    m1 = binvox_rw.read_as_3d_array(f)

a = trimesh.voxel.VoxelGrid(m1.data)
type(a)
with open('chair_out.binvox', 'wb') as f:
    m1.write(f)

with open('chair_out.binvox', 'rb') as f:
    m2 = binvox_rw.read_as_3d_array(f)


with open('chair.binvox', 'rb') as f:
    md = binvox_rw.read_as_3d_array(f)
with open('chair.binvox', 'rb') as f:
    ms = binvox_rw.read_as_coord_array(f)

data_ds = binvox_rw.dense_to_sparse(md.data)
data_sd = binvox_rw.sparse_to_dense(ms.data, 32)


