import numpy as np
import magpylib as magpy
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from stl import mesh

def get_field_pos(data):
    x = np.zeros(shape=(data.shape[0]))
    y = np.zeros(shape=(data.shape[0]))
    z = np.zeros(shape=(data.shape[0]))
    B = np.zeros(shape=(data.shape[0]))
    for n in range(data.shape[0]):
            x[n] = (np.squeeze(data[n, 0]))
            y[n] = (np.squeeze(data[n, 1]))
            z[n] = (np.squeeze(data[n, 2]))
            B[n] = (np.squeeze(data[n, 3]))
    return x, y, z, B



def display_scatter_3D(x, y, z, B, center:bool=False, title:str=None, clim_plot = None):
    
    if center is True:
        x = (x - 0.5 *  np.max(x)) 
        y = (y - 0.5 * np.max(y)) 
        z = (z - 0.5 * np.max(z)) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=B, cmap=plt.jet())
    plt.title(title)
    plt.colorbar(img)
    # cbar = fig.colorbar(img)
    # if clim_plot is None:
    #     cbar.ax.set_ylim(0.95 * np.mean(B),1.05 * np.mean(B))
    # else:
    #     cbar.ax.set_ylim(clim_plot[0],clim_plot[1])
    plt.show()

def scale_wrt_meas(B_eff, scaling_factor):
        weights =  (B_eff - np.min(B_eff)) /(np.max(B_eff) - np.min(B_eff))
        weights2 = 1 - weights
        # scaling_factor2 = 0.5 * scaling_factor
        B_eff_scaled = B_eff * (1 - (weights * 0.15 * scaling_factor)) 
        B_eff_scaled = B_eff_scaled * (1 + (weights2 * scaling_factor))
        return B_eff_scaled


def get_magnetic_field(magnets, sensors, axis = None, scaling_factor = 0.25):
    B = sensors.getB(magnets)
    if axis is None:
        B_eff = np.linalg.norm(B, axis=1)
    else:
        B_eff = np.squeeze(B[:, axis]) 
        B_eff = scale_wrt_meas(B_eff, scaling_factor)
    return B_eff
    
def load_magnets_in_rings(x, shims, num_var, magnetization):
    binary_placement_each_mag=np.array([x[f"x{child:02}"] for child in range(0, num_var * len(shims.children) * len(shims.children[0].children))]) # all children should have same magnet positions to begin with
    shim_ring_optimized = magpy.Collection()
    child_offset = 0
    for child1 in range(len(shims.children)):
        if child1 > 0:
            child_offset += (num_var * len(shims.children[child1-1])) 
        for child2 in range(len(shims.children[child1].children)):
            
            mag = binary_placement_each_mag[child2 + child_offset]
            pol = binary_placement_each_mag[child2 + child_offset + len(shims.children[child1].children)]
            
            if mag == True:
                if pol == True:
                    mag_fact = [magnetization[0], magnetization[1], magnetization[2]]
                    # mag_fact = [0, 6750 * 1e2, 0]
                else:
                    mag_fact = [-magnetization[0], -magnetization[1], -magnetization[2]] # Remove hardcoding TODO
                    # mag_fact = [0, -6750 * 1e2, 0 ] #
                shim_magnet = shims.children[child1].children[child2].copy(magnetization=mag_fact)
                shim_ring_optimized.add(shim_magnet)
                
    return shim_ring_optimized

def load_magnets_in_rings_pol(x, shims):
    binary_placement_each_mag=np.array([x[f"x{child:02}"] for child in range(0, 4 * len(shims.children[0].children))])
    shim_ring_optimized = magpy.Collection()
    
    for child in range(len(shims.children[0].children)):
        mag_upper = binary_placement_each_mag[child]
        pol_upper = binary_placement_each_mag[child + len(shims.children[0].children)]
        
        mag_lower = binary_placement_each_mag[child + 2 * len(shims.children[0].children)]
        pol_lower = binary_placement_each_mag[child + 3 * len(shims.children[0].children)]
        
        if mag_lower == True:
            if pol_lower == True:
                pol_fact = [0, 0 ,6750 * 1e2]
            else:
                pol_fact = [0, 0 ,-6750 * 1e2]# Remove hardcoding TODO
            shim_magnet = shims.children[0].children[child].copy(magnetization = pol_fact)
            shim_ring_optimized.add(shim_magnet)
            
        if mag_upper == True:
            if pol_upper == True:
                pol_fact = [0, 0 ,6750 * 1e2]
            else:
                pol_fact = [0, 0 ,-6750 * 1e2]
            shim_magnet = shims.children[1].children[child].copy(magnetization = pol_fact)
            shim_ring_optimized.add(shim_magnet)
            
    return shim_ring_optimized

def load_magnets_with_rot(x, shims, num_var, magnetization):
    binary_placement_each_mag=np.array([x[f"x{child:02}"] for child in range(0, num_var * len(shims.children) * len(shims.children[0].children))]) # all children should have same magnet positions to begin with
    shim_ring_optimized = magpy.Collection()
    child_offset = 0
    for child1 in range(len(shims.children)):
        if child1 > 0:
            child_offset += (len(shims.children[child1-1])) 
        for child2 in range(len(shims.children[child1].children)):
            
            mag = binary_placement_each_mag[child2 + child_offset]
            rot = binary_placement_each_mag[child2 + child_offset + num_var * len(shims.children[child1].children)]
            
            if mag == True:
                rot_orig = shims.children[child1].children[child2].orientation
                zrot = rot * 180/ np.pi
                rot_pol = R.from_euler('zyx',[0, 0, zrot],degrees=True)
                rot_total = rot_orig * rot_pol
                shim_magnet = shims.children[child1].children[child2].copy(orientation = rot_total) #
                shim_ring_optimized.add(shim_magnet)
                
    return shim_ring_optimized

def filter_dsv(x, y, z, B, dsv_radius):
    x_new = []
    y_new = []
    z_new = []
    B_new = []
    
    for point in range(x.shape[0]):
        dist = np.sqrt (x[point] ** 2 + y[point] **2 + z[point]** 2)
        if dist < dsv_radius:
            x_new.append(x[point])
            y_new.append(y[point])
            z_new.append(z[point])
            B_new.append(B[point])
            
    x_new = np.array(x_new)
    y_new = np.array(y_new)
    z_new = np.array(z_new)
    B_new = np.array(B_new)
    
    return x_new, y_new, z_new, B_new

def cost_fn(B_total):
    f = 1e3 * (np.max(B_total) - np.min(B_total)) /(np.mean(B_total))
    return f

def write2stl(mag_collection_template, stl_filename:str='output.stl'):
    # Get the trace of each cuboid
    init = 0
    if (len(mag_collection_template.children)) > 2: # TODO: better condition check single collection, no multiple trays
        for child1 in range(len(mag_collection_template.children)): 
            cube_magnet = mag_collection_template.children[child1]
            mag_cuboid_trace = mag_collection_template.children[child1].get_trace()
            vertices = np.array([mag_cuboid_trace['x'], mag_cuboid_trace['y'], mag_cuboid_trace['z']]).T + cube_magnet.orientation.apply(cube_magnet.position.T)
            faces = np.array([mag_cuboid_trace['i'], mag_cuboid_trace['j'], mag_cuboid_trace['k']]).T
            vertices_rotated = cube_magnet.orientation.apply(vertices)
            cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    cube_mesh.vectors[i][j] = vertices_rotated[f[j]] * 1e3 # because stl printing is in mm
            if init > 0:
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh.data]))
            else:
                init = 1
                cube_mesh_all = cube_mesh
    else:
        for child1 in range(len(mag_collection_template.children)):   
            for child2 in range(len(mag_collection_template.children[child1].children)): # assumes more than one plate
                cube_magnet = mag_collection_template.children[child1].children[child2]
                mag_cuboid_trace = mag_collection_template.children[child1].children[child2].get_trace()

                vertices = np.array([mag_cuboid_trace['x'], mag_cuboid_trace['y'], mag_cuboid_trace['z']]).T + cube_magnet.orientation.apply(cube_magnet.position.T)
                faces = np.array([mag_cuboid_trace['i'], mag_cuboid_trace['j'], mag_cuboid_trace['k']]).T
                vertices_rotated = cube_magnet.orientation.apply(vertices)
                cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        cube_mesh.vectors[i][j] = vertices_rotated[f[j]] * 1e3
                if init > 0:
                    cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh.data]))
                else:
                    init = 1
                    cube_mesh_all = cube_mesh
        
    # Save the STL file
    cube_mesh_all.save(stl_filename)
             
            
    
    # # Perform triangulation
    # collection_no_pol = load_magnets_with_rot(X, mag_collection_template, 2 ,magnetization=magnetization)
    # collection_no_pol.show()
    # # Subtract the magnet locations from a cylinder of 1.5 times the thickness of the magnet
    # print('To be implemented')
        