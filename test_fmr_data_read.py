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
import matplotlib.pyplot as plt
import pickle
import time

# Changelog:
# 2024-08-21 - sairamgeethanath: Reflecting support for new fmr mapping axes - aligned with magpy
# 2026-08-21 - sairamgeethanath: Data variable is now an array of shape (N, 5) with columns x, y, z, B, V. The get_field_pos function has been updated to handle this new format.


#---------------------------------------------------------
# Read magnetic field and positions
#---------------------------------------------------------
fname = './data/Exp_1033_2026824.npy'
data = np.load(fname)
resolution = 4 #mm
x, y, z, B, V, dx, dy, dz = get_field_pos(data)
print(x.shape, y.shape, z.shape, B.shape, V.shape)

plt.plot(B, color='blue')
plt.xlabel('Index')
plt.ylabel('B (mT)')
plt.title('Measured B Field')
plt.show()

plt.plot(V, color='orange')
plt.xlabel('Index')
plt.ylabel('V (mT)') 
plt.title('Measured V value')
plt.show()

r = np.sqrt(dx**2 + dy**2 + dz**2)

plt.plot(r, color='green')
plt.xlabel('Index')
plt.ylabel('Distance (mm)')
plt.title('Distance between two measurements')
plt.show()