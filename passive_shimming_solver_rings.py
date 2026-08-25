import numpy as np
import magpylib as magpy
from make_shim_rings import make_shim_ring_template
from utils import get_field_pos, display_scatter_3D, get_magnetic_field, load_magnets_with_rot, filter_dsv, cost_fn, write2stl, _write_tray
from colorama import Style, Fore
from target_B0_2_shim_locations_rot import shimming_problem
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import pickle
import time
import trimesh

# Changelog:
# 2024-08-21 - sairamgeethanath: Reflecting support for new fmr mapping axes - aligned with magpy
# 2026-08-21 - sairamgeethanath: Data variable is now an array of shape (N, 5) with columns x, y, z, B, V. The get_field_pos function has been updated to handle this new format.
# 2026-08-25 - sairamgeethanath: Added functionality to separate the optimized shim tray into top and bottom trays, save them as STL files, and visualize them with distinct colors.

#---------------------------------------------------------
# Shim plate dimensions
#---------------------------------------------------------
disk_dia = 6 * 2.54 * 1e-2 # in meter
disk_thickness = 3.2 * 1e-3 # in meter

#---------------------------------------------------------
# Filenames
#---------------------------------------------------------
fmr_data_fname = './data/Exp_1033_2026824.npy'
stl_filename = './data/shim_tray_template_dia_20260825_'
pkl_filename = './data/magnet_collection_shims_20260825.pkl'
fname_top_tray = './data/shim_tray_top_20260825.stl'
fname_bottom_tray = './data/shim_tray_bottom_20260825.stl'

#---------------------------------------------------------
# Read magnetic field and positions
#---------------------------------------------------------

data = np.load(fmr_data_fname)
resolution = 4 #mm
gammabar = 42.577478518e6 # Hz/T
x, y, z, B, V, dx, dy, dz = get_field_pos(data)
# Apply the 4 mT correction only at positions where x > -12 mm.
# B = np.where(x > -12, B - 4, B)


# custom filtering based on the error of 4mT
x_magpy = x  * 1e-3 #conversion to m
y_magpy = y * 1e-3 #conversion to m
z_magpy = z * 1e-3 #conversion to m
B = B * 1e-3 # mT to T


# Display measured field as scattered data - plot3
vmin = 0.266
vmax = 0.270
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False, title = 'Measured B field', vmin = vmin, vmax = vmax)
print(Fore.RED + 'Delta B0 before shimming: ' + str((np.max(B) - np.min(B)) * 1e3) + 'mT')
print(Fore.RED + 'D B0 before shimming: ' + str(np.mean(B) * 1e3) + 'mT')
print(Fore.CYAN + 'Off-resonance before shimming is:' + str((np.max(B) - np.min(B)) * gammabar * 1e-3) + ' kHz') # What decimal should we round off to? 1mT - 85kHz
pos = np.zeros((x.shape[0], 3))
pos[:, 0] = x_magpy
pos[:, 1] = y_magpy
pos[:, 2] = z_magpy

dsv_sensors = magpy.Collection(style_label='sensors')
sensor1 = magpy.Sensor(position=pos,style_size=2)
dsv_sensors.add(sensor1)
print(Fore.GREEN + 'Done creating position sensors')

#---------------------------------------------------------
# Specify geometry of the shim array - biplanar
#---------------------------------------------------------
magnet_dims_x =  6.35 *1e-3 # m
magnet_dims_y =   6.35 *1e-3 # m
magnet_dims_z =   3.18 *1e-3 # m
diameter = 152.4 * 1e-3 # m
mag_x = 0
mag_z = 8 * 1e5
mag_y = 0
magnetization = [mag_x, mag_y, mag_z] # 1.34, 0.7957 
# heights = np.array([-41.325, 41.325]) * 1e-3
# heights = np.array([-150.325, 150.325]) * 1e-3
heights = np.array([-36.10, 36.10]) * 1e-3
num_magnets = 300 # 100
delta_B0_tol = 0.5 * 1e-3 # Tesla 

# Create lower shim tray
shim_rings_template_stl = make_shim_ring_template(diameter, magnet_dims = (magnet_dims_x, magnet_dims_y, magnet_dims_z), 
                                              heights = [heights[0]], num_magnets=num_magnets, magnetization=magnetization, symmetry = False,
                                              style_color='red')
shim_rings_template_stl.show(backend='matplotlib')

write2stl(shim_rings_template_stl, stl_filename =stl_filename+str(diameter * 1e3)+ '.stl', debug=False)
shim_rings_template = make_shim_ring_template(diameter, magnet_dims = (magnet_dims_x, magnet_dims_y, magnet_dims_z), 
                                              heights = heights, num_magnets=num_magnets, magnetization=magnetization, symmetry = False,
                                              style_color='red')
shim_rings_template.show(backend='matplotlib')
# write2stl(shim_rings_template, stl_filename ='./data/init10_arrangement_symm_dia_'+str(diameter * 1e3)+ '.stl', debug=True)
magpy.show(shim_rings_template, dsv_sensors)


#
B0_computed = get_magnetic_field(magnets=shim_rings_template, sensors=dsv_sensors, axis = 2)
vmin_shim = -0.015
vmax_shim = 0.015
display_scatter_3D(x_magpy, y_magpy, z_magpy, B0_computed, center=False, title = 'B computed from shim tray template', vmin = vmin_shim, vmax = vmax_shim)
display_scatter_3D(x_magpy, y_magpy, z_magpy, B0_computed + B, center=False, title = 'B + B0_computed', vmin = np.mean(B0_computed + B) - 0.5 * 1e-3, vmax = np.mean(B0_computed + B) + 0.5 *1e-3)

#---------------------------------------------------------
# Solve for the homogeneity constraints using an optimization problem - explore constraints, free geometry in a single plane, etc.
#---------------------------------------------------------

print(Fore.YELLOW + 'Shim search starts ...')
del_B_init = np.mean(B) - B
pop_size = 500 # Size of the population
shim_trays_optimize = shimming_problem(B_measured=B, tol=delta_B0_tol, 
                                       shims=shim_rings_template, sensors=dsv_sensors,
                                       num_var=2, magnetization=magnetization)

algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
tic = time.time()
res = minimize(shim_trays_optimize,
                algorithm, ('n_gen', 200),
                verbose=True)
toc = time.time()
print(Fore.YELLOW + 'Shim search ends ...')
print(Fore.YELLOW + 'Time taken for optimization:' + str(toc - tic) + 's')
# Get the locations where the magnets need to be present and make a new collection

shim_rings_optimized = load_magnets_with_rot(res.X, shim_rings_template, 2 ,magnetization=magnetization, style_color='green')
shim_rings_optimized.show()


#  ---------------------------------------------------------
# STL file generation and field computation
#  ---------------------------------------------------------
write2stl(shim_rings_optimized, stl_filename = stl_filename + 'shims_slots.stl')
B_shimmed = get_magnetic_field(shim_rings_optimized, dsv_sensors, axis = 2)
B_total = B + B_shimmed 

print(Fore.CYAN + 'Off-resonance indicator after shimming is:' + str(cost_fn(B_total)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
display_scatter_3D(x_magpy, y_magpy, z_magpy, B_total, center=False, title='B0 after shimming', vmin = vmin, vmax = vmax)
print(Fore.RED + 'Delta B0_shimmed: ' + str((np.max(B_total) - np.min(B_total)) * 1e3) + 'mT')

# Get the shimmed field and show a subplot of measured and shimmed field
print(Fore.RED + 'Done shimming!')

# ---------------------------------------------------------
# Save the optimized shim tray
# ---------------------------------------------------------
with open(pkl_filename, 'wb') as file:
    pickle.dump(shim_rings_optimized, file)
# Figure how to export this to CAD
# Check if the shim tray can be loaded and displayed

with open(pkl_filename, 'rb') as file:
    shim_rings_optimized_read = pickle.load(file)
shim_rings_optimized_read.show()


# ---------------------------------------------------------
# Prepare to save individual trays and with marks on them 
# ---------------------------------------------------------
#---------------------------------------------------------
# Visualization colors
# RGBA values: [R, G, B, alpha]
# These affect visualization only, not STL geometry.
#---------------------------------------------------------
TOP_TRAY_COLOR = [110, 180, 230, 255]       # light blue
BOTTOM_TRAY_COLOR = [240, 180, 90, 255]     # light orange


# Count the number of magnets in the optimized shim tray
num_magnets = len(shim_rings_optimized_read)
print(
    Fore.GREEN
    + 'Number of magnets in the optimized shim tray: '
    + str(num_magnets)
)

# Separate the magnets into two collections based on their z position
shim_rings_top = magpy.Collection(
    style_label='top_shims'
)

shim_rings_bottom = magpy.Collection(
    style_label='bottom_shims'
)

num_magnets = 0

for magnet in shim_rings_optimized_read:

    original_position = np.array(
        magnet.position,
        copy=True
    )

    stored_position = original_position.copy()
    stored_position[2] = 0

    cuboid = magpy.magnet.Cuboid(
        magnetization=np.array(
            magnet.magnetization,
            copy=True
        ),
        dimension=np.array(
            magnet.dimension,
            copy=True
        ),
        position=stored_position,
        orientation=magnet.orientation,
    )

    if original_position[2] > 0:
        shim_rings_top.add(cuboid)
        num_magnets += 1

    else:
        shim_rings_bottom.add(cuboid)
        num_magnets += 1


print(
    Fore.GREEN
    + 'Total number of magnets in the optimized shim tray: '
    + str(num_magnets)
)

print(
    Fore.GREEN
    + 'Number of magnets in the top shim ring: '
    + str(len(shim_rings_top))
)

print(
    Fore.GREEN
    + 'Number of magnets in the bottom shim ring: '
    + str(len(shim_rings_bottom))
)


#---------------------------------------------------------
# Visualize the top and bottom shim rings separately
#---------------------------------------------------------

# Set visualization colors for the magnetic cuboids.
# This does not alter position, orientation, dimensions, or magnetization.
for magnet in shim_rings_top:
    magnet.style.color = '#6EB4E6'

for magnet in shim_rings_bottom:
    magnet.style.color = '#F0B45A'


shim_rings_top.show()
shim_rings_bottom.show()

#---------------------------------------------------------
# Write STL files
#---------------------------------------------------------

_write_tray(
    shim_rings_top,
    fname_top_tray,
    1
)

_write_tray(
    shim_rings_bottom,
    fname_bottom_tray,
    -1
)


#---------------------------------------------------------
# Reload the exported trays and display them
# for visual inspection.
#---------------------------------------------------------

top_tray_mesh = trimesh.load_mesh(
    'shim_tray_top.stl'
)

bottom_tray_mesh = trimesh.load_mesh(
    'shim_tray_bottom.stl'
)


#---------------------------------------------------------
# Apply visualization colors.
#
# These colors exist only in the Trimesh viewer.
# They do NOT change the STL geometry.
#---------------------------------------------------------

top_tray_mesh.visual.face_colors = (
    TOP_TRAY_COLOR
)

bottom_tray_mesh.visual.face_colors = (
    BOTTOM_TRAY_COLOR
)


#---------------------------------------------------------
# Display trays
#---------------------------------------------------------

trimesh.Scene(
    {
        'top_tray':
            top_tray_mesh
    }
).show()


trimesh.Scene(
    {
        'bottom_tray':
            bottom_tray_mesh
    }
).show()
