import numpy as np
import magpylib as magpy
from make_shim_rings import make_shim_ring_template
from utils import get_field_pos, display_scatter_3D, get_magnetic_field, load_magnets_in_rings
from colorama import Style, Fore
from target_B0_2_shim_locations_v2 import shimming_problem
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
import matplotlib.pyplot as plt

# Read magnetic field and positions
fname = './fmr_data/calibration_jig/calibration_data.npy'
data = np.load(fname)
resolution = 2 #mm
x, y, z, B = get_field_pos(data)
print(Fore.GREEN + 'Done reading data')
x = (np.float64(x).transpose() - 0.5 * np.max(x))  * 1e-3 #conversion to m
y = (np.float64(y).transpose() - 0.5 * np.max(y)) * 1e-3 #conversion to m
z = (np.float64(z).transpose() - 0.5 * np.max(z)) * 1e-3 #conversion to m
B = B * 1e-3 # mT to T

# Map robot space to magpy space
x_magpy = z # length
y_magpy = x # depth
z_magpy = -y # height



# Display measured field as scattered data - plot3
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False)
print(Fore.CYAN + 'Mean B0 before shimming is:' + str(np.round(np.mean(B),2)) + ' mT') # What decimal should we round off to? 1mT - 85kHz
pos = np.zeros((x.shape[0], 3))
pos[:, 0] = x_magpy
pos[:, 1] = y_magpy
pos[:, 2] = z_magpy

dsv_sensors = magpy.Collection(style_label='sensors')
sensor1 = magpy.Sensor(position=pos,style_size=2)
dsv_sensors.add(sensor1)
print(Fore.GREEN + 'Done creating position sensors')

# Specify geometry of the magnet calibration jig
magnet_dims_x = 6.35 *1e-3 # m
magnet_dims_y = 6.35 *1e-3 # m
magnet_dims_z = 6.35 *1e-3 * 0.5 # m
magnet_dims = (magnet_dims_x, magnet_dims_y, magnet_dims_z)
diameter = 60 * 1e-3 # m
magnetization = [0, 0 ,8750 * 1e2] # 1.34, 0.7957
position1 = np.multiply([0, 0, 8 + (0.5 * magnet_dims_z)], 1e-3)

# Make the jig 
steps = 100
mag_vect = np.linspace(6000*1e2,9000*1e2,steps)
B_diff =[]
for ind in range(len(mag_vect)):
    cube = magpy.magnet.Cuboid(
                            dimension=magnet_dims,
                            # polarization=self.polarization,
                            position=position1,
                            # polarization=polarization,
                            magnetization=[0, 0 ,mag_vect[ind]],
                            style_color='red',
                        )

    B_sim = get_magnetic_field(cube, dsv_sensors, axis = 2)
    B_diff.append(np.linalg.norm(B - B_sim))
    
    # display_scatter_3D(x_magpy, y_magpy, z_magpy, B_sim, center=False)
best_fit = np.where(B_diff == np.min(np.abs(B_diff)))
print(best_fit)
print(mag_vect[best_fit])
    # Figure how to export this to CAD

    # plt.plot(B, label = 'Measured')
    # plt.plot(B_sim, label = 'Simulated')
    # plt.legend()
    # plt.show()

    # Visualize differences
    # display_scatter_3D(x_magpy, y_magpy, z_magpy, (B - B_sim), center=False)



