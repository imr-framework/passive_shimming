
# target_B0_2_shim_locations
# The problem can be defined as:
# Find the combination of children in the magnet collections that when used provide the most homogeneous magnetic field with a given tolerance
# 
import magpylib as magpy
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from utils import get_magnetic_field, load_magnets_with_rot, cost_fn

class shimming_problem(ElementwiseProblem):
   
    def __init__(self, B_measured, tol, shims, sensors, num_var, magnetization, **kwargs):
        self.tol = tol
        self.shims = shims
        self.sensors = sensors
        self.B_measured = B_measured
        self.B_mean = np.mean(B_measured)
        self.B_del_init = self.B_measured - self.B_mean
        self.B_std= np.std(B_measured)
        self.num_var = num_var
        self.magnetization = magnetization
        vars = dict()
        num_slots = len(self.shims.children) *  len(self.shims.children[0].children)
        for child in range(0, num_slots): # binary choice if the magnet is going to be on or off - for the upper and lower rings so 2 * len
            vars[f"x{child:02}"] = Binary()
        for child in range(num_slots, 2 * num_slots): #2nd variable - remove hardcoding TODO
            vars[f"x{child:02}"] = Choice(options=list(np.r_[0:2*np.pi:np.pi/2]))
        self.x = vars
        
        super().__init__(vars=vars, n_ieq_constr=0, n_obj=1, **kwargs)
    
    def _evaluate(self, x, out, *args, **kwargs):
        
        self.shim_ring_optimized = load_magnets_with_rot(x, self.shims, self.num_var, self.magnetization)
        if (len(self.shim_ring_optimized)==0):
           self.B_shimmed = 0
        else:
            self.B_shimmed = get_magnetic_field(self.shim_ring_optimized, self.sensors, axis = 2)
        # f1 = np.linalg.norm(self.B_mean - (self.B_measured + self.B_shimmed))
        # del_f = 42580 * np.std(self.B_measured + self.B_shimmed)  # can minimize both difference and std. explicitly
        B_total = self.B_measured + self.B_shimmed

        # del_B = np.max(B_total) - np.min(B_total)
        # f1 = del_B * 1e3 #mT
        # g1 = del_f - 50e3 #self.B_std 
        
        # Minimize global off resonance 
        # f1 = 42580 * np.std(B_total) # kHz
        
        # Minimize range of B and maximize mean
        f1 = cost_fn(B_total)
        out["F"] = [f1]
        # out["G"] = g1
    
        


    
            