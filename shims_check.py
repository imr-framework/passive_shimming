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
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('Qt5Agg')

# Read magnetic field and positions
fname = './data/Tenacity_36_2mm.npy'
data = np.load(fname)
resolution = 2 #mm
x, y, z, B = get_field_pos(data)
print(Fore.GREEN + 'Done reading data')
x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
B = B * 1e-3 # mT to T
print('Mean value of B:' + str(np.mean(B)))
dsv_radius = 16 * 1e-3 # m
x, y, z, B = filter_dsv(x, y, z, B, dsv_radius = dsv_radius)

B_orig = B
# Map robot space to magpy space
x_magpy = z # length
y_magpy = x # depth
z_magpy = -y # height

# Display measured field as scattered data - plot3
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False, title = 'Measured B field')
display_scatter_3D(x_magpy, y_magpy, z_magpy, B - np.mean(B), center=False, title = 'Measured B field - mean sub')
print(Fore.RED + 'del B0: ' + str((np.max(B) - np.min(B)) * 1e3) + 'mT')
print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(np.round(cost_fn(B),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
pos = np.zeros((x.shape[0], 3))
pos[:, 0] = x_magpy
pos[:, 1] = y_magpy
pos[:, 2] = z_magpy

dsv_sensors = magpy.Collection(style_label='sensors')
sensor1 = magpy.Sensor(position=pos,style_size=2)
dsv_sensors.add(sensor1)
print(Fore.GREEN + 'Done creating position sensors')

# Specify geometry of the shim array - biplanar
magnet_dims_x =  6.35 *1e-3 # m
magnet_dims_y =  6.35 *1e-3 # m
magnet_dims_z =  6.35 *1e-3 # m
diameter = 60 * 1e-3 # m
mag_x = 0
mag_z = 8 * 1e5
mag_y = 0
magnetization = [mag_x, mag_y, mag_z] # 1.34, 0.7957
heights = np.array([-41.325, 41.325]) * 1e-3
num_magnets = 30
delta_B0_tol = 1 * 1e-3 # Tesla

# Create lower shim tray
shim_rings_template = make_shim_ring_template(diameter, magnet_dims = (magnet_dims_x, magnet_dims_y, magnet_dims_z), 
                                              heights = heights, num_magnets=num_magnets, magnetization=magnetization)
shim_rings_template.show(backend='matplotlib')



# Figure how to export this to CAD
with open('./data/magnet_collection_shims.pkl', 'rb') as file:
    shim_rings_optimized_read = pickle.load(file)
shim_rings_optimized_read.show(background=True, backend='matplotlib')


B0_computed = get_magnetic_field(magnets=shim_rings_optimized_read, sensors=dsv_sensors, axis = 2)
B_total = B0_computed + B
display_scatter_3D(x_magpy, y_magpy, z_magpy, B0_computed, center=False, title = 'B computed from optimized shim tray')
display_scatter_3D(x_magpy, y_magpy, z_magpy, B_total, center=False, title = 'B + B0_computed')
plt.show()
print('Mean value of B_shimmed:' + str(np.mean(B_total)))
print(Fore.RED + 'del B0: ' + str((np.max(B_total) - np.min(B_total)) * 1e3) + 'mT')
print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(np.round(cost_fn(B_total),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz

print('Percentage reduction in inhomogeneity:' + str(100 - np.round((np.max(B_total) - np.min(B_total)) / (np.max(B) - np.min(B)) * 100, 2)) + '%')

# Visualize the B field from the shim trays once they are mapped
shim_field_mapped = True
if shim_field_mapped is True:
    fname_shim = './data/shim_tray_32_2mm_new.npy'
    data_shim = np.load(fname_shim)
    resolution = 2 #mm
    do_threshold = True
    x, y, z, B_shim = get_field_pos(data_shim)
    print(Fore.GREEN + 'Done reading data')
    x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
    y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
    z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
    B_shim = -1 * B_shim * 1e-3 # mT to T
    print('Mean value of B shim:' + str(np.mean(B_shim)))
    dsv_radius = 16 * 1e-3 # m
    x, y, z, B_shim = filter_dsv(x, y, z, B_shim, dsv_radius = dsv_radius)

    if do_threshold is True:
        B_shim[np.abs(B_shim) > 1] = 0

    # Map robot space to magpy space
    x_magpy = z # length
    y_magpy = x # depth
    z_magpy = -y # height

    # Display measured field as scattered data - plot3
    display_scatter_3D(x_magpy, y_magpy, z_magpy, B_shim, center=False, title = 'Measured B shim field')
    display_scatter_3D(x_magpy, y_magpy, z_magpy, B + B_shim, center=False, title = 'Predicted total field - mapped')
    print(Fore.RED + 'del B0: ' + str((np.max(B + B_shim) - np.min(B + B_shim)) * 1e3) + 'mT')
    print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(np.round(cost_fn(B_shim + B),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz

compute_shim_diff = True
if compute_shim_diff:
        fname_shim = './data/Tenacity_shimmed.npy'
        data = np.load(fname_shim)
        resolution = 2 #mm
        x, y, z, B_shimmed = get_field_pos(data)
        print(Fore.GREEN + 'Done reading data')
        x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
        y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
        z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
        B_shimmed = B_shimmed * 1e-3 # mT to T
        print('Mean value of B_shimmed:' + str(np.mean(B)))
        x, y, z, B_shimmed = filter_dsv(x, y, z, B_shimmed, dsv_radius = dsv_radius)
        # Map robot space to magpy space
        x_magpy = z # length
        y_magpy = x # depth
        z_magpy = -y # height
        
        print(Fore.RED + 'del B0: ' + str((np.max(B_shimmed) - np.min(B_shimmed)) * 1e3) + 'mT')
        print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(np.round(cost_fn(B_shimmed),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
        
        display_scatter_3D(x_magpy, y_magpy, z_magpy, B_shimmed, center=False, title = 'Measured B shimmed field')
        display_scatter_3D(x_magpy, y_magpy, z_magpy, B_shimmed - B_orig, center=False, title = 'B shimmed - B original')
    
        diff = B_shimmed - B_orig
        plt.plot(diff)
        plt.show()
        