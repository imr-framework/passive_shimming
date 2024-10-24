import numpy as np
import magpylib as magpy
from make_shim_rings import make_shim_ring_template
from utils import get_field_pos, display_scatter_3D, get_magnetic_field, load_magnets_in_rings, filter_dsv
from colorama import Style, Fore
import matplotlib.pyplot as plt

# Read magnetic field and positions
fname = './data/Characterization in the office/Exp_3 ( 2 haifs of the shim tray), 30 30 30 2.npy'
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
display_scatter_3D(x_magpy, y_magpy, z_magpy, B, center=False, title='Measured field')
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
magnetization = [0, 0 , 1320] # 1.34, 0.7957
position1 = np.multiply([0, 0, 8 + (0.5 * magnet_dims_z)], 1e-3)
position2 = np.multiply([0, 0, -8 - (0.5 * magnet_dims_z)], 1e-3)

# Make the jig 
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
                                              heights = heights, num_magnets=num_magnets, magnetization=magnetization, skip_center_magnets=True)
shim_rings_template.show(backend='matplotlib')

B_sim = get_magnetic_field(shim_rings_template, dsv_sensors, axis = 2)
display_scatter_3D(x_magpy, y_magpy, z_magpy, B_sim, center=False, title = 'Simulated B field')

# Figure how to export this to CAD
plt.plot(B, label = 'Measured')
plt.plot(B_sim, label = 'Simulated')
plt.legend()
plt.show()

# Visualize differences
display_scatter_3D(x_magpy, y_magpy, z_magpy, (B / B_sim), center=False)

plt.plot(B / B_sim, label = 'Simulated')
plt.legend()
plt.show()



