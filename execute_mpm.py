import taichi as ti
import numpy as np
import h5py
import time
import os
from mpm_solver import MPM_solver


ti.init(arch = ti.gpu)
simulator = MPM_solver(3, 10) # whatever, will be re-initialized

simulator.load_from_sampling("/initial_pois_sampling.h5")
simulator.initialize_parameters()
simulator.add_bounding_box()
# simulator.add_surface_collider(point=(0, 0.1, 0),normal=(0, 1, 0))
simulator.add_surface_leader(point=(0.25,0,0), normal=(1,0,0), v_BC=(0.2, 0, 0), threshold = simulator.dx)
simulator.add_surface_leader(point=(0,0,0), normal=(1,0,0), v_BC=(0, 0, 0), threshold = simulator.dx)

# a surface_leader is a plane defined by one point on the plane and its normal. All grid nodes that are within threshold
# of the plane will have v set to v_BC. Next step, point = point + dt * v_BC

gui = ti.GUI("Explicit MPM", res=768, background_color=0x112F41)
colors = np.array([0xED553B,0x068587,0xEEEEF0], dtype=np.uint32)
gui.circles(simulator.x_2d.to_numpy(), radius=1.5, color=colors[simulator.material.to_numpy()])
gui.show()


for frame in range(0,10):
    simulator.p2g2p(frame, 0.005)
    simulator.save_data_at_frame("../simulation_data", frame, flag_p2d = True, save_to_h5 = True, save_to_obj = True)
    gui.circles(simulator.x_2d.to_numpy(), radius=1.5, color=colors[simulator.material.to_numpy()])
    gui.show()
