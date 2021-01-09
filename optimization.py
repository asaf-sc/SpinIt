import trimesh
import numpy as np
import binvox_rw
import slqp

from results import get_results
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
    x, z, y= shell.matrix.nonzero()
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
    global p
    s1 = mass_properties["mass"]
    sx, sz, sy = mass_properties["center_mass"]/s1 -p
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

def s_singlebox_shifted(mass_properties):
    global p
    s1 = mass_properties["mass"]
    sx, sz, sy = mass_properties["center_mass"]/s1

    Icom =  mass_properties["inertia"]
    I = Icom + ((sz*sz)/s1)*np.diag([1,1,0])

    sxy = -I[0][1]
    syz = -I[1][2]
    sxz = -I[1][2]
    sx2 = (-I[0][0] +I[1][1] +I[2][2])/2
    sy2 = (I[0][0] -I[1][1] +I[2][2])/2
    sz2 =(I[0][0] +I[1][1] -I[2][2])/2
    return [s1, sx, sy, sz, sxy, syz, sxz, sx2, sy2, sz2]


def laplacian():
    global relevant_voxels, map_index_to_num
    x,z,y = relevant_voxels.nonzero()
    indices = list(zip(x,y,z))
    laplacian = np.zeros((len(x), len(x)))
    map_index_to_num = {key: value for (value, key) in enumerate(indices)}
    x_lim, z_lim, y_lim = relevant_voxels.shape
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

def get_origin(shell):
    matrix = shell.matrix[:,0,:]
    return matrix.nonzero()[0][0],0, matrix.nonzero()[1][0]


def f_yoyo (gamma_I, Ia, Ib, Ic):
    return gamma_I*(pow((Ia/Ic),2)+ pow((Ib/Ic),2))

def f_top (f_yoyo, gamma_c, l, M):
    return gamma_c*pow(l*M, 2)+f_yoyo


if __name__ == '__main__':
    pitch = 0.04
    relevant_voxels = None
    map_index_to_num = None
    # filename = 'ellipsoid2_3.obj'
    # filename = 'dolphin.obj'
    # filename = 'kitten1_4.obj'
    filename = 'kitten1_4_2.binvox'
    filename = 'kitten1_4_7.binvox' #20
    # filename = 'kitten1_4_6.binvox' #12
    # filename = 'kitten1_4_4.binvox' # 100
    # filename = 'kitten1_4_5.binvox' # 50
    # filename = 'ellipsoid2_3_1.binvox'
    # filename = 'rabbit2_3.binvox'
    # filename = 'ellipsoid2_3_5.binvox' #6
    # filename = 'ellipsoid2_3_6.binvox' #20

    with open(filename, 'rb') as f:
        binvox_obj = binvox_rw.read_as_3d_array(f)

    # obj = trimesh.load_mesh(filename)
    # voxel_obj = voxelize_and_fill(obj)
    voxel_obj = trimesh.voxel.VoxelGrid(binvox_obj.data)
    mass_properties = get_voxel_obj_mass_properties(voxel_obj)
    center_mass = mass_properties["center_mass"]
    mass = mass_properties["mass"]
    inertia = mass_properties["inertia"]
    shell = trimesh.voxel.VoxelGrid(binvox_obj.data).hollow()
    relevant_voxels = voxel_obj.matrix ^ shell.matrix
    p = get_origin(shell)

    # axis = trimesh.creation.axis(origin_color=[1., 0, 0])
    # scene1 = trimesh.Scene()
    # # scene.add_geometry(voxel_obj_filled)
    # scene1.add_geometry(axis)
    # g = trimesh.voxel.VoxelGrid(voxel_obj.matrix.swapaxes(1, 2))
    # scene1.add_geometry(g)
    # scene1.show()

    s = get_s(mass_properties)
    s_shell = compute_s_shell()
    laplacian = laplacian()
    s_omega_k_matrix = compute_s() #doesnt change anymore
    s_everything = s_shell + np.sum(s_omega_k_matrix, axis=0) #doesnt change anymore
    k = np.sum(relevant_voxels)
    beta = np.ones(k)

    #input to optimization problem: s_omega_omega_tag
    beta_k = slqp.slqp(laplacian,s_everything, s_omega_k_matrix)
    # beta_k = get_result()

    map_index_to_num_inv = dict([(value, key) for key, value in map_index_to_num.items()])
    result_mat = voxel_obj.matrix.copy()

    for index in np.round(beta_k).nonzero()[0]:
        x, y, z = map_index_to_num_inv[index]
        result_mat[x, z, y] = False
    result = trimesh.voxel.VoxelGrid(result_mat)
    # result.show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:,:,:5])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:,:,5:])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:,5:,:])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:,:5,:])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[:5,:,:])).show()
    trimesh.voxel.VoxelGrid(np.asarray(result_mat[5:,:,:])).show()
    print("Results:")
    print(result.as_boxes().mass_properties)
    print("voxel_obj:")
    print(voxel_obj.as_boxes().mass_properties)
    print("p:")
    print(p)
    # slice.show()
    result_mesh = result.as_boxes()
    result_mesh.visual.face_colors = [100, 100, 100, 100]  # make transparent
    result_mesh.show()
    trimesh.voxel.VoxelGrid(np.asarray([result_mat[:, :, 3]])).show()

    print("h")
    # [0.9796104025296949, 1.0, 0.9334090914248037, 0.968126688076059, 1.0, 1.0, 0.9212563432072022, 0.959219185050817, 0.9972150793869922, 1.0, 0.892834555298958, 0.948220183856354, 0.9915340743016836, 0.819456789397245, 0.0, 0.3929258688309422, 0.4914655548383254, 0.1484546374457337, 0.2807077402047117, 0.4106872548244752, 0.09759939824175227, 0.1918739183920986, 0.33564869519937524, 0.11571053695201508, 0.9774152896726088, 0.9923150967887113, 1.0, 1.0, 1.0, 0.9657507438188474, 0.9879539148180838, 1.0, 1.0, 0.9565165626609738, 0.9828903657436232, 1.0, 1.0, 0.9455266542499917, 0.9770540353481455, 1.0, 1.0, 0.43092742687722585, 0.5795401326644813, 0.7714481526703817, 0.9321433148330025, 0.34255421439862144, 0.469801386291618, 0.6554639397177479, 0.8610691144120985, 0.21797512477548045, 0.35358328650857346, 0.5504001535091211, 0.7751931905984832, 0.13832857423349637, 0.25857519700020787, 0.45474723804000944, 0.7058468381723213, 0.20741775749029237, 0.3568772967970297, 0.992935963113994, 1.0, 1.0, 1.0, 1.0, 0.9901639577723185, 1.0, 1.0, 1.0, 0.9859194498908779, 1.0, 1.0, 1.0, 0.9806125567355156, 0.9982039576776058, 1.0, 1.0, 1.0, 0.8099560013382474, 0.9299182255427506, 1.0, 1.0, 0.28919624323021775, 0.42593229343759764, 0.6442985344031942, 0.8384414437946202, 0.983610809703966, 1.0, 0.21289599210655286, 0.33343900194439297, 0.5197308078877145, 0.7350766429508139, 0.9278651094393893, 1.0, 0.11173632486782335, 0.21966528544810213, 0.3998181617781251, 0.6299907100468172, 0.8606261545233576, 1.0, 0.03363385742221517, 0.12693741079069287, 0.29619157941630575, 0.5332888599857178, 0.7996622420852811, 1.0, 0.08225268974724054, 0.22817327613861627, 0.4596186261730111, 0.7681370619186716, 0.9779393342705239, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9981296136438929, 1.0, 1.0, 1.0, 1.0, 0.9947960037363415, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9528534463221999, 1.0, 1.0, 1.0, 0.6276622851974628, 0.8084625971777756, 0.9455762943538811, 1.0, 1.0, 1.0, 0.2964329222466158, 0.46412851057632437, 0.6675814515568881, 0.8631673137367529, 1.0, 1.0, 1.0, 0.21156563913846407, 0.33886706358848445, 0.5380196736324622, 0.7625212121790491, 0.9502703796496819, 1.0, 1.0, 0.10633142826517142, 0.21789450095021268, 0.41260547641509554, 0.6550820289756449, 0.8852884254894929, 1.0, 1.0, 0.022432856089164, 0.11624107432804405, 0.29810324613203865, 0.548758528279848, 0.8192171000216912, 1.0, 0.0, 0.05532549334917925, 0.20213043969184938, 0.4398537353677928, 0.7634610023203952, 0.9721339859739084, 0.12191297674805807, 0.24903918585544038, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9296529567601447, 1.0, 1.0, 1.0, 0.612494904751733, 0.7923216165417141, 0.93856510447812, 1.0, 1.0, 1.0, 0.2808785740321694, 0.45298358470515604, 0.6551615520422815, 0.8541230576748815, 0.9955571030071829, 1.0, 1.0, 0.1900143908879748, 0.31952038749748446, 0.5212458255982699, 0.7494266767204125, 0.9417715192251539, 1.0, 1.0, 0.08553138063863731, 0.19580451862478837, 0.3919264222346997, 0.638035567382673, 0.8731350785952259, 1.0, 1.0, 0.008346050150496937, 0.09379583609067049, 0.2728082956567576, 0.5285970835646671, 0.7996383140871312, 0.9845620265407882, 0.0, 0.0320439337848781, 0.16851324012726937, 0.4366991561714892, 0.7384544587051978, 0.9459539559086556, 0.056775057758428095, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.907779135904964, 0.988541870404556, 1.0, 1.0, 0.5867556912838916, 0.7624040045034363, 0.9136761815604779, 1.0, 1.0, 1.0, 0.25174866110860405, 0.4180331889881248, 0.6180995930548568, 0.8197314657592096, 0.9742314619721231, 1.0, 1.0, 0.1469303801182909, 0.27720953104089086, 0.47549185774911157, 0.7024168717318446, 0.9056988161889822, 1.0, 1.0, 0.05230920687505247, 0.15594353516392045, 0.3439228329774513, 0.584390917058124, 0.8283224685155826, 0.9963339925531272, 1.0, 0.0, 0.06220777555261097, 0.2236622626666605, 0.46488128422663916, 0.7430014194693825, 0.9446800423364309, 0.006256149867909695, 0.11737066318412051, 0.33630701387398526, 0.6746489490106304, 0.8951194791639039, 0.029745338372962993, 0.11759275037178339, 0.7315304986489486, 0.8700308810453851, 0.9947023924574371, 1.0, 0.37664932724751277, 0.5760922492594333, 0.7756396857988912, 0.9440997547952008, 1.0, 1.0, 0.08063995730035192, 0.2292689462610644, 0.415052694661227, 0.6262435068912208, 0.8338333558141361, 1.0, 1.0, 0.02120156458514235, 0.11691827723689742, 0.28790027204662205, 0.5060297631109186, 0.7445438119644675, 0.9604882912265887, 0.04119487323908652, 0.17145797292313253, 0.38547762048894696, 0.6578988682202687, 0.894764702085553, 0.0, 0.06940412198208493, 0.2578038028251747, 0.6114165361846021, 0.8445733550517757, 0.0, 0.08098073389803263, 0.3585174584280088, 0.5091712060748509, 0.6669973954373623, 0.26800760440605964, 0.4230946055071747, 0.5960603161096415, 0.14998020743757837, 0.30515095625587013, 0.5184559023314566, 0.03765534324056381, 0.15259134665167137, 0.0, 0.0]