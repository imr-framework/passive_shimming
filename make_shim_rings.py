# Make shim rings 
import numpy as np
import magpylib as magpy

class shim_ring:
    # initiates an instance of the shim ring
    def __init__(self, diameter, magnet_dims, height, num_magnets, magnetization):
        self.diameter = diameter
        self.magnet_dims = magnet_dims
        self.height = height
        self.num_magnets = num_magnets
        self.display_collection = False
        self.magnetization = magnetization
        
    def get_geometry_params(self):
        # self.num_rings = int(self.diameter / (np.sqrt(self.magnet_dims[0] ** 2  +  self.magnet_dims[1] ** 2)) / 2)
        self.num_rings = int(self.diameter / (2 * (np.sqrt(self.magnet_dims[0] ** 2  + self.magnet_dims[1] ** 2))))
        
        # print('Number of rings in this shim tray:' + str(self.num_rings))
        # get the radius for each ring with the first ring radius = magnet_dims[1]
        
    def make_magnet_collection(self):
        shim_ring_magnets = magpy.Collection()
        total_num_magnets_ring=0
        for r in range(self.num_rings):
            if r == 0:
                cube = magpy.magnet.Cuboid(
                dimension=self.magnet_dims,
                position=(0 + (self.magnet_dims[0] * 0.5),0,self.height),
                magnetization=self.magnetization
                )
                shim_ring_magnets.add(cube)
                
                cube = magpy.magnet.Cuboid(
                dimension=self.magnet_dims,
                # magnetization=self.magnetization,
                position=(0 - (self.magnet_dims[0] * 0.5),0,self.height),
                magnetization=self.magnetization
                )
                shim_ring_magnets.add(cube)
                num_magnets_ring = 2
                total_num_magnets_ring += num_magnets_ring
            else:
                radius = r * (np.sqrt(self.magnet_dims[0] ** 2  +  self.magnet_dims[1] ** 2))
                semi_circumference = np.pi * radius
                num_magnets_ring = int(1.5 * np.floor(semi_circumference / self.magnet_dims[0]))
                # num_magnets_ring = int(2 * radius / self.magnet_dims[0])
                total_num_magnets_ring += num_magnets_ring
                angles = np.linspace(0, 360, num_magnets_ring, endpoint=False)
                # "print %d" % (3.78) 
                for a in angles:
                    cube = magpy.magnet.Cuboid(
                        dimension=self.magnet_dims,
                        position=(radius,0,self.height),
                        magnetization=self.magnetization,
                        style_color='red',
                    )
                    cube.rotate_from_angax(a, 'z', anchor=0)
                    cube.rotate_from_angax(a, 'z')
                    shim_ring_magnets.add(cube)
            print ("Ring %d has %d elements" % (r, num_magnets_ring))
        print("Total magnets required for this shim tray:" + str(total_num_magnets_ring))
        self.collection = shim_ring_magnets
        if self.display_collection is True:
            shim_ring_magnets.show(backend='matplotlib')
            
def make_shim_ring_template(diameter, magnet_dims, heights, num_magnets, magnetization):
    shim_rings_template = magpy.Collection(style_label='magnets')
    for height in heights:
        shim_ring_ind = shim_ring(diameter=diameter, magnet_dims=magnet_dims, height =height,
                            num_magnets = num_magnets, magnetization = magnetization)
        shim_ring_ind.get_geometry_params() # can improve this to make it more flexible and dynamic - include update methods TODO
        shim_ring_ind.make_magnet_collection()

        # Add both shim trays as one collection
        
        shim_rings_template.add(shim_ring_ind.collection)
    return shim_rings_template
    
if __name__ == "__main__":
    height = -45
    shim_ring_upper = shim_ring(diameter=127, magnet_dims=(6.35,6.35,3.175 * 3) * 1e-3, height =height,
                                num_magnets = 500)
    shim_ring_upper.get_geometry_params()
    shim_ring_upper.make_magnet_collection()
    # shim_ring_lower = make_shim_ring()





