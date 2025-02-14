# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 16:32:31 2025

@author: mn1215
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import integrate

from src.methods.image_point import ImagePointAdvector 
from src.solvers.explicit import ForwardEulerSolver
from src.systems.lorenz63 import Lorenz63

if __name__ == "__main__":
    
    # Initialise the Lorenz 63 system - 'high resolution'.
    sys = Lorenz63()
        
    # Solver parameters.
    endtime = 100
    dt = 0.01
    nt = int(endtime/dt)
    solver = ForwardEulerSolver(sys.force, endtime, nt)
    
    # Run solver starting from initial condition. 
    x0 = np.array([-4.32, -6.0, 18.34])
    solver.run(x0)
    
    # Plot the trajectory.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].plot(solver.xs[:, 0], solver.xs[:, 2], '-k', linewidth=0.2)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Z")
    
    # Plot the vector field.
    x, y, z = solver.xs.T
    force = sys.force(solver.xs)
    axs[1].quiver(x, z, force[:, 0], force[:, 2], color='k', alpha=0.7, width=0.005)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    plt.show()
    
    #%% IMAGE POINT WITHOUT NUDGING 
    
    # Initialise the image point advector.
    N = 10
    x_ref = solver.xs.copy()
    vector_field = sys.force(x_ref)
    advector = ImagePointAdvector(x_ref, vector_field, N)
    
    # Setup the solver.
    endtime = 200
    dt = 0.01
    nt = int(endtime/dt)
    solver_adv = ForwardEulerSolver(advector.force, endtime, nt)
    
    # Run the solver.
    y0 = x0.copy()
    solver_adv.run(y0)
        
    # Subplots.
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(solver_adv.xs[:, 0], solver_adv.xs[:, 2], 'k-', linewidth=0.2)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend(["Image Point"])
    
    axs[1].plot(advector.x_ref[:, 0], advector.x_ref[:, 2], 'k-', linewidth=0.2)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    axs[1].legend(["Ref"])
    plt.show()
    
    #%% IMAGE POINT WITH NUDGING 
    
    # Initialise the image point advector with nudging.
    N = 10
    advector = ImagePointAdvector(x_ref, vector_field, N, eta=1, nudge=True)
    
    # Setup the solver.
    endtime = 200
    dt = 0.01
    nt = int(endtime/dt)
    solver_adv = ForwardEulerSolver(advector.force, endtime, nt)
    
    # Run the solver.
    y0 = x0.copy()
    solver_adv.run(y0)
        
    # Subplots.
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(solver_adv.xs[:, 0], solver_adv.xs[:, 2], 'k-', linewidth=0.2)
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend(["Image Point"])
    
    axs[1].plot(advector.x_ref[:, 0], advector.x_ref[:, 2], 'k-', linewidth=0.2)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Z")
    axs[1].legend(["Ref"])
    plt.show()
    