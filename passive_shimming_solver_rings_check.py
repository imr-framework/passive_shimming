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

# Read magnetic field and positions
# fname = './data/Shim_trays_30mm_dsv.npy'
# fname = '/Volumes/SD_US/fmr/bat x-tion/Exp_3_2024927.npy'
fname = './data/Exp_100_2024917.npy'
data = np.load(fname)
resolution = 2 #mm
x, y, z, B = get_field_pos(data)
print(Fore.GREEN + 'Done reading data')
x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
B = B * 1e-3 # mT to T

dsv_radius = 15 * 1e-3 # m
x, y, z, B = filter_dsv(x, y, z, B, dsv_radius = dsv_radius)

# Map robot space to magpy space
x_magpy = z # length
y_magpy = x # depth
z_magpy = -y # height

# Display measured field as scattered data - plot3
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False, title = 'Measured B field')
display_scatter_3D(x_magpy, y_magpy, z_magpy, B - np.mean(B), center=False, title = 'Measured B field - mean sub')
print(Fore.RED + 'del B0: ' + str((np.max(B) - np.min(B)) * 1e3) + 'mT')
print(Fore.GREEN + 'Mean B0: ' + str((np.mean(B)) * 1e3) + 'mT')
print(Fore.CYAN + 'Off-resonance indicator before shimming is:' + str(cost_fn(B)) + 'DelB/B * 1000') # What decimal should we round off to? 1mT - 85kHz
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
                                              heights = heights, num_magnets=num_magnets, magnetization=magnetization,skip_center_magnets=False)
shim_rings_template.show(backend='matplotlib')
write2stl(shim_rings_template, stl_filename ='./data/init10_arrangement.stl')
magpy.show(shim_rings_template, dsv_sensors)


#
B0_computed = get_magnetic_field(magnets=shim_rings_template, sensors=dsv_sensors, axis = 2)
display_scatter_3D(x_magpy, y_magpy, z_magpy, B0_computed, center=False, title = 'B computed from shim tray template')

# Figure how to export this to CAD
plt.plot(B, label = 'Measured')
plt.plot(B0_computed, label = 'Simulated')
plt.legend()
plt.show()