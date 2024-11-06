import numpy as np
import magpylib as magpy
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from stl import mesh
import math

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



def display_scatter_3D(x, y, z, B, center:bool=False, title:str=None, clim_plot = None, vmin = 0.265, vmax = 0.271):
    
    if center is True:
        x = (x - 0.5 *  np.max(x)) 
        y = (y - 0.5 * np.max(y)) 
        z = (z - 0.5 * np.max(z)) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x, y, z, c=B, cmap='coolwarm', vmin = vmin, vmax = vmax) #coolwarm
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

def load_magnets_with_rot(x, shims, num_var, magnetization, style_color = 'green'):
    binary_placement_each_mag=np.array([x[f"x{child:02}"] for child in range(0, num_var * len(shims.children) * len(shims.children[0].children))]) # all children should have same magnet positions to begin with
    shim_ring_optimized = magpy.Collection()
    child_offset = 0
    num_trays = len(shims.children)
    for child1 in range(len(shims.children)):
        if child1 > 0:
            child_offset += (len(shims.children[child1-1])) 
        for child2 in range(len(shims.children[child1].children)):
            
            mag = binary_placement_each_mag[child2 + child_offset]
            rot = binary_placement_each_mag[child2 + child_offset + num_trays * len(shims.children[child1].children)] #TODO: check for consistency
            
            if mag == True:
                rot_orig = shims.children[child1].children[child2].orientation
                zrot = rot * 180/ np.pi
                rot_pol = R.from_euler('zyx',[0, 0, zrot],degrees=True)
                rot_total = rot_orig * rot_pol
                shim_magnet = shims.children[child1].children[child2].copy(orientation = rot_total) #
                shim_magnet.style_color = style_color
                shim_ring_optimized.add(shim_magnet)
                
    return shim_ring_optimized

def filter_dsv(x, y, z, B, dsv_radius, symmetry = True):
    x_new = []
    y_new = []
    z_new = []
    B_new = []
    
    for point in range(x.shape[0]):
        if symmetry:
            if x[point] >= 0 and y[point] >= 0 and z[point] >= 0:
                dist = np.sqrt (x[point] ** 2 + y[point] **2 + z[point]** 2)
                if dist < dsv_radius:
                    x_new.append(x[point])
                    y_new.append(y[point])
                    z_new.append(z[point])
                    B_new.append(B[point])
        else:
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

def write2stl(mag_collection_template, stl_filename:str='output.stl', debug = False):
    # Get the trace of each cuboid
    init = 0
    faces = 0
    if (len(mag_collection_template.children)) > 2: # TODO: better condition check single collection, no multiple trays
        for child1 in range(len(mag_collection_template.children)): 
            cube_magnet = mag_collection_template.children[child1]
            mag_cuboid_trace = mag_collection_template.children[child1].get_trace()
            vertices = np.array([mag_cuboid_trace['x'], mag_cuboid_trace['y'], mag_cuboid_trace['z']]).T + cube_magnet.position.T  #cube_magnet.orientation.apply(cube_magnet.position.T)
            
            x = vertices[:, 0]
            y = vertices[:, 1]
            
            # if np.min(x) < 0:
            #     print('Negative x')
            #     print(x)
            #     x+= np.abs(np.min(x))
            # if np.min(y) < 0:
            #     print('Negative y')
            #     print(y)
            #     y+= 2 * np.abs(np.min(y))
            
            
            faces = np.array([mag_cuboid_trace['i'], mag_cuboid_trace['j'], mag_cuboid_trace['k']]).T
            
            rot_angle = math.radians(cube_magnet.orientation.as_euler('xyz', degrees=True)[2])
            rot_matrix = np.array([
                    [math.cos(rot_angle), -math.sin(rot_angle), 0],
                    [math.sin(rot_angle), math.cos(rot_angle), 0],
                    [0, 0, 1]
                ])
            vertices_rotated = np.dot(vertices, rot_matrix)  # because stl printing is in mm 
            vertices_flipped_x = vertices_rotated * [-1, 1, 1]
            vertices_flipped_y = vertices_rotated * [1, -1, 1]
            vertices_flipped_xy = vertices_rotated * [-1, -1, 1]

            cube_mesh =  make_mesh(faces, vertices_rotated)
            cube_mesh_flipped_x = make_mesh(faces, vertices_flipped_x)
            cube_mesh_flipped_y = make_mesh(faces, vertices_flipped_y)
            cube_mesh_flipped_xy = make_mesh(faces, vertices_flipped_xy)
                    
            if init > 0:
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh.data]))
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh_flipped_x.data]))
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh_flipped_y.data]))
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh_flipped_xy.data]))
            else:
                init = 1 
                cube_mesh_all = cube_mesh
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh_flipped_x.data]))
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh_flipped_y.data]))
                cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh_flipped_xy.data]))
            if debug:
                    cube_mesh_all.save(stl_filename)
    else:
        for child1 in range(len(mag_collection_template.children)):   
            for child2 in range(len(mag_collection_template.children[child1].children)): # assumes more than one plate
                cube_magnet = mag_collection_template.children[child1].children[child2]
                mag_cuboid_trace = mag_collection_template.children[child1].children[child2].get_trace()

                vertices = np.array([mag_cuboid_trace['x'], mag_cuboid_trace['y'], mag_cuboid_trace['z']]).T  #+
                faces = np.array([mag_cuboid_trace['i'], mag_cuboid_trace['j'], mag_cuboid_trace['k']]).T
                vertices_rotated = cube_magnet.orientation.apply(vertices) * 1e3 # because stl printing is in mm
        
                if debug:
                    rot_matrix = cube_magnet.orientation.as_matrix()
                    euler_angles = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True) 
                    print(euler_angles)
                    print(cube_magnet.position)

                        
                cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        cube_mesh.vectors[i][j] = vertices_rotated[f[j]]
                        
                if init > 0:
                    cube_mesh_all = mesh.Mesh(np.concatenate([cube_mesh_all.data, cube_mesh.data]))
                else:
                    init = 1
                    cube_mesh_all = cube_mesh
                if debug:
                    cube_mesh_all.save(stl_filename)
                
     
    # Save the STL file
    cube_mesh_all.save(stl_filename)
             
            
    
    # # Perform triangulation
    # collection_no_pol = load_magnets_with_rot(X, mag_collection_template, 2 ,magnetization=magnetization)
    # collection_no_pol.show()
    # # Subtract the magnet locations from a cylinder of 1.5 times the thickness of the magnet
    # print('To be implemented')
        
def undo_symmetry_8x_compression(mag_collection):
    mag_collection_uncompressed = magpy.Collection()
    for child in range(len(mag_collection.children)):
            # 0
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([1, 1, 1])
            shim_magnet.style_color = 'blue'
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            # mag_collection_uncompressed.show(backend='matplotlib')
            
            # 90 degrees
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([-1, 1, 1])
            rot_pol = R.from_euler('zyx',[ 0, 0, 90],degrees=True)
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            # mag_collection_uncompressed.show(backend='matplotlib')
            
            
            # 180 degrees
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([-1, -1, 1])
            rot_pol = R.from_euler('zyx',[ 0, 0, 180],degrees=True)
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            # mag_collection_uncompressed.show(backend='matplotlib')
            
            # 270 degrees
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([1, -1, 1])
            rot_pol = R.from_euler('zyx',[ 0, 0, 270],degrees=True)
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            # mag_collection_uncompressed.show(backend='matplotlib')
            
            # ----------------------------------------------------------------
            # Upper plate
            # ----------------------------------------------------------------
            # 0
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([1, 1, -1])
            shim_magnet.style_color = 'blue'
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            
            # 90 degrees - upper plate
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([-1, 1, -1])
            rot_pol = R.from_euler('zyx',[0, 90, 0],degrees=True)
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            # mag_collection_uncompressed.show(backend='matplotlib')
            
            # 180 degrees
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([-1, -1, -1])
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            
            # 270 degrees
            shim_magnet = mag_collection.children[child].copy()
            shim_magnet.position = shim_magnet.position * np.array([1, -1, -1])
            # shim_magnet.orientation = shim_magnet.orientation * rot_pol
            mag_collection_uncompressed.add(shim_magnet)
            
            
            
    return mag_collection_uncompressed

def visualize_shim_tray(shim_trays, tray:str='upper'):
    if tray == 'upper':
        shim_tray_upper = magpy.Collection()
        for child in range(len(shim_trays)):
            cube_magnet = shim_trays[child].copy()
            if cube_magnet.position[2] >= 0:
                cube_magnet.style_color = 'green'
                shim_tray_upper.add(cube_magnet)
        shim_tray_chosen = shim_tray_upper
       
    elif tray == 'lower':
        shim_tray_lower = magpy.Collection()
        for child in range(len(shim_trays.children[0].children)):
            cube_magnet = shim_trays.children[0].children[child].copy()
            if cube_magnet.position[2] < 0:
                cube_magnet.style_color = 'blue'
                shim_tray_lower.add(cube_magnet)
        shim_tray_chosen = shim_tray_lower

    shim_tray_chosen.show(backend='matplotlib')
    return shim_tray_chosen

def make_mesh(faces, vertices_rotated):
# # ---------------------------------------------------------
# # Open saved optimized shim tray
# convert this into a function definition once it works
    cube_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube_mesh.vectors[i][j] = vertices_rotated[f[j]] * 1e3 # because stl printing is in mm
            
    return cube_mesh