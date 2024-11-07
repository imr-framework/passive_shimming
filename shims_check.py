import numpy as np
import magpylib as magpy
from make_shim_rings import make_shim_ring_template
from utils import get_field_pos, display_scatter_3D, get_magnetic_field, filter_dsv, cost_fn, undo_symmetry_8x_compression,write2stl, visualize_shim_tray
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
#---------------------------------------------------------
# Read magnetic field and positions from field mapping robot
#---------------------------------------------------------
fname = './data/Exp_21_20241012.npy'
data = np.load(fname)
resolution = 4 #mm
x, y, z, B = get_field_pos(data)
print(Fore.GREEN + 'Done reading data')
x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
B = B * 1e-3 # mT to T
print('Mean value of B:' + str(np.mean(B)))
dsv_radius = 30 * 1e-3 # m
x, y, z, B = filter_dsv(x, y, z, B, dsv_radius = dsv_radius, symmetry=False)

B_orig = B
# Map robot space to magpy space
x_magpy = z # length
y_magpy = x # depth
z_magpy = -y # height

# Display measured field as scattered data - plot3
delB = 0.002 # T
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False, title = 'Measured B field', vmin = 0.265, vmax = 0.2695)
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

#---------------------------------------------------------
# Open saved optimized shim tray
# ---------------------------------------------------------
# fname_pkl = './data/magnet_collection_shims_6inch_60mag.pkl'
fname_pkl = './data/magnet_collection_shims_pf2.pkl'
with open(fname_pkl, 'rb') as file:
    shim_rings_optimized_read = pickle.load(file)
write2stl(shim_rings_optimized_read, stl_filename ='./data/optimized_arrangement_dia_'+str(152.4 * 1e3)+ '.stl', debug=False)
shim_rings_optimized_read.show(background=True, backend='matplotlib')


# ---------------------------------------------------------
# Undo the symmetry compression 
#---------------------------------------------------------
shim_rings_optimized_uncompressed = undo_symmetry_8x_compression(shim_rings_optimized_read)
shim_rings_optimized_uncompressed.show(background=True, backend='matplotlib')
shim_tray_single = visualize_shim_tray(shim_rings_optimized_uncompressed, tray='upper')
# write2stl(shim_tray_single, stl_filename ='./data/optimized_arrangement_dia_'+str(152.4 * 1e3)+ '.stl', debug=False)
shim_tray_single.show(background=True, backend='matplotlib')
# ---------------------------------------------------------
# Compute the B field from the optimized shim tray
# ---------------------------------------------------------

B_predicted_shim = get_magnetic_field(magnets=shim_rings_optimized_uncompressed, sensors=dsv_sensors, axis = 2)
B_total_predicted = B_predicted_shim + B
display_scatter_3D(x_magpy, y_magpy, z_magpy, B_predicted_shim, center=False, title = 'B computed from optimized shim tray', 
                   vmin = -0.0015, vmax = 0.001)
display_scatter_3D(x_magpy, y_magpy, z_magpy, B_total_predicted, center=False, title = 'B total predicted', vmin = 0.265 , vmax = 0.2695)
plt.show()
print('Mean value of predicted B_shimmed:' + str(np.mean(B_total_predicted)))
print(Fore.RED + 'Del B0 post shimming: ' + str((np.max(B_total_predicted) - np.min(B_total_predicted)) * 1e3) + 'mT')
print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(np.round(cost_fn(B_total_predicted),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz

print('Percentage reduction in inhomogeneity:' + str(100 - np.round((np.max(B_total_predicted) - np.min(B_total_predicted)) / (np.max(B) - np.min(B)) * 100, 2)) + '%')

# ---------------------------------------------------------
# Visualize the B field from the shim trays once they are mapped
# ---------------------------------------------------------
shim_field_mapped = True
if shim_field_mapped is True:
    fname_shim = './data/Exp_23_20241012.npy'
    data_shim = np.load(fname_shim)
    resolution = 4 #mm
    do_threshold = True
    x, y, z, B_shim = get_field_pos(data_shim)
    print(Fore.GREEN + 'Done reading data')
    x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
    y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
    z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m/
    B_shim = B_shim * 1e-3 # mT to T
    dsv_radius = 16 * 1e-3 # m
    x, y, z, B_shim = filter_dsv(x, y, z, B_shim, dsv_radius = dsv_radius, symmetry=False)
    print('Mean value of B shim:' + str(np.mean(B_shim)))


    # Map robot space to magpy space
    x_magpy = z # length
    y_magpy = x # depth
    z_magpy = -y # height
    print(Fore.RED + 'Del B0 post shimming: ' + str((np.max(B_shim) - np.min(B_shim)) * 1e3) + 'mT')
    print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(np.round(cost_fn(B_shim),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
    print('Percentage reduction in inhomogeneity:' + str(100 - np.round((np.max(B_shim) - np.min(B_shim)) / (np.max(B) - np.min(B)) * 100, 2)) + '%')

    # Display measured field as scattered data - plot3
    display_scatter_3D(x_magpy, y_magpy, z_magpy, B_shim, center=False, title = 'Measured B shimmed field', vmin = 0.265, vmax = 0.2695)
 
compute_shim_diff = False
if compute_shim_diff:
        fname_shim = './data/Exp_15_2024109.npy'
        data = np.load(fname_shim)
        resolution = 4 #mm
        x, y, z, B_shimmed = get_field_pos(data)
        print(Fore.GREEN + 'Done reading data')
        x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
        y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
        z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
        B_shimmed = B_shimmed * 1e-3 # mT to T
        print('Mean value of B_shimmed:' + str(np.mean(B_shimmed)))
        x, y, z, B_shimmed = filter_dsv(x, y, z, B_shimmed, dsv_radius = dsv_radius)
        # Map robot space to magpy space
        x_magpy = z # length
        y_magpy = x # depth
        z_magpy = -y # height
        
        print(Fore.RED + 'del B0: ' + str((np.max(B_shimmed) - np.min(B_shimmed)) * 1e3) + 'mT')
        print(Fore.CYAN + 'Off-resonance indicator after shimming is:' + str(np.round(cost_fn(B_shimmed),2)) + ' DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
        
        display_scatter_3D(x_magpy, y_magpy, z_magpy, B_shimmed, center=False, title = 'Measured B shimmed field')
        display_scatter_3D(x_magpy, y_magpy, z_magpy, B_shimmed - B_orig, center=False, title = 'B shimmed - B original')
    
        diff = B_shimmed - B_orig
        plt.plot(diff)
        plt.show()
        