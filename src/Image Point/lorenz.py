# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:28:39 2025

@author: mn1215
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt

class Lorenz63:
    
    @dataclass
    class Parameters:
        """ 
        Container class for storing the parameters of the Lorenz 63 system."""
                
        sigma: float = 10.
        beta: float = 10/3
        rho: float = 28.
    
    def __init__(self):

        self.params = self.Parameters()
    
    def force(self, x: np.ndarray) -> np.ndarray:
        """ 
        Evaluates the right-hand-side (force vector) of the Lorenz 63 system."""
        
        # Check for correct dimension.        
        single_point = False 
        if x.ndim == 1:
            single_point = True 
            x = x.reshape(1, 3)
        
        dxdt = self.params.sigma*(x[:, 1]-x[:, 0])
        dydt = x[:, 0]*(self.params.rho - x[:, 2]) - x[:, 1]
        dzdt = x[:, 0]*x[:, 1] - self.params.beta*x[:, 2]
        
        f = np.column_stack((dxdt, dydt, dzdt))
                
        return f[0] if single_point else f

class ForwardEulerSolver:
    
    def __init__(self, model: Callable[[np.ndarray], np.ndarray], endtime: float, nt: float):
        """ 
        Args: 
            model: system of equations that will be iterated.
            endtime: total simulation runtime.
            nt: no. of time steps."""
        
        self.model = model
        self.endtime = endtime 
        self.nt = nt 
        
        self.store = True
        self.xs = None
        
    def run(self, x0: np.ndarray) -> None:
        
        # Calculate timestep.
        dt = self.endtime/self.nt
        
        # Initialise storage arrays.
        if self.store:
            self.xs = np.zeros(shape=(self.nt, 3))
            self.xs[0] = x0
            
        for t in range(1, self.nt):
            
            # Progress the system.
            x = x0 + self.model(x0)*dt
            
            # Update old value and store.
            x0 = x
            if self.store:
                self.xs[t] = x0

class ImagePointAdvector:

    def __init__(self, x_ref: np.ndarray, model: Callable[[np.ndarray], np.ndarray], 
                 N: int, eta: Optional[float]=None, nudge: Optional[bool]=False):
        """ 
        Args: 
            x_ref: reference solution (high-res or observations)
            model: function that returns the rhs of pde.
            N: no. of points in the neihbourhood to calculate force.
            
            eta: nudging parameter (default None)
            nudge: on/off flag for nudging."""
        
        self.N = N
        self.x_ref = x_ref
        
        # Nudging parameters.
        self.eta = eta
        self.nudge = nudge 
        
    def force(self, y: np.ndarray) -> np.ndarray:
        """ 
        Calculates the average force over the neighbourhood of y."""
        
        # Find the N nearest neighbours to y.
        x_neighbours = self._nearest_neighbours(y)
        
        # Average force in the neighbourhood.
        f_avg = np.sum(sys.force(x_neighbours), axis=0)/self.N
        
        if self.nudge and self.eta is not None:
            nudge_term = self.eta*np.sum(x_neighbours - y, axis=0)/self.N
            return f_avg + nudge_term
        
        return f_avg
    
    def _nearest_neighbours(self, y: np.ndarray) -> np.ndarray:
        """ 
        Finds the N closest points of x_ref to y."""
        
        # Calculates l2 norm and returns points that minimise it.
        norm = np.linalg.norm(y - self.x_ref, 2, axis=1)
        return self.x_ref[np.argsort(norm)[:self.N]]
    

#%%
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
    
    #%% Advection of the image point.
        
    # Initialise the image point advector.
    N = 10
    advector = ImagePointAdvector(solver.xs, sys.force, N)
    
    # Setup the solver.
    endtime = 0.5
    dt = 0.01
    nt = int(endtime/dt)
    solver_adv = ForwardEulerSolver(advector.force, endtime, nt)
    
    # Run the solver.
    y0 = x0.copy()
    solver_adv.run(y0)
    
    # Plot this trajectory.
    plt.figure(figsize=(8, 8))
    plt.plot(advector.x_ref[:, 0], advector.x_ref[:, 2], 'k-', linewidth=0.2)
    plt.plot(solver_adv.xs[:, 0], solver_adv.xs[:, 2], 'or', markersize=2)
    plt.show()
        
    #%% Advection of the image point with nudging.
    
    # Initialise the image point advector with nudging.
    N = 10
    advector = ImagePointAdvector(solver.xs, sys.force, N, eta=1, nudge=True)
    
    # Setup the solver.
    endtime = 0.5
    dt = 0.01
    nt = int(endtime/dt)
    solver_adv = ForwardEulerSolver(advector.force, endtime, nt)
    
    # Run the solver.
    y0 = x0.copy()
    solver_adv.run(y0)
    
    # Plot this trajectory.
    plt.figure(figsize=(8, 8))
    plt.plot(advector.x_ref[:, 0], advector.x_ref[:, 2], 'k-', linewidth=0.2)
    plt.plot(solver_adv.xs[:, 0], solver_adv.xs[:, 2], 'or', markersize=2)
    plt.show()