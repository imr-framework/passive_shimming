import numpy as np
import magpylib as magpy
from make_shim_rings import make_shim_ring_template
from utils import get_field_pos, display_scatter_3D, get_magnetic_field, load_magnets_with_rot, filter_dsv, cost_fn, write2stl
from colorama import Style, Fore
from target_B0_2_shim_locations_rot import shimming_problem
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import pickle

#---------------------------------------------------------
# Read magnetic field and positions
#---------------------------------------------------------
fname = './data/Exp_21_20241012.npy'
data = np.load(fname)
resolution = 2 #mm
x, y, z, B = get_field_pos(data)
print(Fore.GREEN + 'Done reading data')
x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
B = B * 1e-3 # mT to T

dsv_radius = 32 * 1e-3 # m
x, y, z, B = filter_dsv(x, y, z, B, dsv_radius = dsv_radius, symmetry = True)

# Map robot space to magpy space
x_magpy = z # length
y_magpy = x # depth
z_magpy = -y # height

# Display measured field as scattered data - plot3
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False, title = 'Measured B field')
display_scatter_3D(x_magpy, y_magpy, z_magpy, B - np.mean(B), center=False, title = 'Measured B field - mean sub')
print(Fore.RED + 'Delta B0 before shimming: ' + str((np.max(B) - np.min(B)) * 1e3) + 'mT')
print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(cost_fn(B)) + 'DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
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
magnet_dims_z =  0.5 *  6.35 *1e-3 # m
diameter = 152.4 * 1e-3 # m
mag_x = 0
mag_z = 8 * 1e5
mag_y = 0
magnetization = [mag_x, mag_y, mag_z] # 1.34, 0.7957 
heights = np.array([-41.325, 41.325]) * 1e-3
# heights = np.array([-150.325, 150.325]) * 1e-3
num_magnets = 100
delta_B0_tol = 1 * 1e-3 # Tesla 

# Create lower shim tray
shim_rings_template_stl = make_shim_ring_template(diameter, magnet_dims = (magnet_dims_x, magnet_dims_y, magnet_dims_z), 
                                              heights = [heights[0]], num_magnets=num_magnets, magnetization=magnetization, symmetry = False,
                                              style_color='red')
shim_rings_template_stl.show(backend='matplotlib')

write2stl(shim_rings_template_stl, stl_filename ='./data/init10_arrangement_single_dia_'+str(diameter * 1e3)+ '.stl', debug=False)
shim_rings_template = make_shim_ring_template(diameter, magnet_dims = (magnet_dims_x, magnet_dims_y, magnet_dims_z), 
                                              heights = heights, num_magnets=num_magnets, magnetization=magnetization, symmetry = True,
                                              style_color='red')
shim_rings_template.show(backend='matplotlib')
# write2stl(shim_rings_template, stl_filename ='./data/init10_arrangement_symm_dia_'+str(diameter * 1e3)+ '.stl', debug=True)
# magpy.show(shim_rings_template, dsv_sensors)


#
B0_computed = get_magnetic_field(magnets=shim_rings_template, sensors=dsv_sensors, axis = 2)
display_scatter_3D(x_magpy, y_magpy, z_magpy, B0_computed, center=False, title = 'B computed from shim tray template')
display_scatter_3D(x_magpy, y_magpy, z_magpy, B0_computed + B, center=False, title = 'B + B0_computed')

#---------------------------------------------------------
# Solve for the homogeneity constraints using an optimization problem - explore constraints, free geometry in a single plane, etc.
#---------------------------------------------------------

print(Fore.YELLOW + 'Shim search starts ...')
del_B_init = np.mean(B) - B
pop_size = 100 # Size of the population
shim_trays_optimize = shimming_problem(B_measured=B, tol=delta_B0_tol, 
                                       shims=shim_rings_template, sensors=dsv_sensors,
                                       num_var=2, magnetization=magnetization)

algorithm = MixedVariableGA(pop_size=pop_size, survival=RankAndCrowdingSurvival())
res = minimize(shim_trays_optimize,
                algorithm, ('n_gen', 20),
                verbose=True)

# Get the locations where the magnets need to be present and make a new collection

shim_rings_optimized = load_magnets_with_rot(res.X, shim_rings_template, 2 ,magnetization=magnetization, style_color='green')
shim_rings_optimized.show()


#  ---------------------------------------------------------
# STL file generation and field computation
#  ---------------------------------------------------------
write2stl(shim_rings_optimized, stl_filename ='./data/shims_slots.stl')
B_shimmed = get_magnetic_field(shim_rings_optimized, dsv_sensors, axis = 2)
B_total = B + B_shimmed 

print(Fore.CYAN + 'Off-resonance indicator after shimming is:' + str(cost_fn(B_total)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
display_scatter_3D(x_magpy, y_magpy, z_magpy, B_total, center=False, title='B0 after shimming')
print(Fore.RED + 'Delta B0_shimmed: ' + str((np.max(B_total) - np.min(B_total)) * 1e3) + 'mT')

# Get the shimmed field and show a subplot of measured and shimmed field
print(Fore.RED + 'Done shimming!')

# ---------------------------------------------------------
# Save the optimized shim tray
# ---------------------------------------------------------
with open('./data/magnet_collection_shims.pkl', 'wb') as file:
    pickle.dump(shim_rings_optimized, file)
# Figure how to export this to CAD
# Check if the shim tray can be loaded and displayed
with open('./data/magnet_collection_shims.pkl', 'rb') as file:
    shim_rings_optimized_read = pickle.load(file)
shim_rings_optimized_read.show()