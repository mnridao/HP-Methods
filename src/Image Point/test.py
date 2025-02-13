# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:32:22 2025

@author: mn1215
"""

from dataclasses import dataclass
from typing import Sequence, Callable

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
    
class Grid3D:
    """ 
    Class responsible for creating a 3D rectangular grid."""
    
    def __init__(self, xbounds: Sequence[float], ybounds: Sequence[float], zbounds: Sequence[float], 
                 nx: int, ny: int, nz: int):
        
        self.xbounds = xbounds 
        self.ybounds = ybounds 
        self.zbounds = zbounds
        
        self.nx = nx 
        self.ny = ny 
        self.nz = nz
        
        self._setup_grid()
        
    def _setup_grid(self):
        
        # Calculate the grid spacings.
        self.dx = self._grid_spacing(self.xbounds, self.nx)
        self.dy = self._grid_spacing(self.ybounds, self.ny)
        self.dz = self._grid_spacing(self.zbounds, self.nz)
        
        # Create 1D array of points in each dimension.
        xpoints = self._points(self.xbounds, self.nx+1)
        ypoints = self._points(self.ybounds, self.ny+1)
        zpoints = self._points(self.zbounds, self.nz+1)
        
        # Create 3D mesh.
        self.x, self.y, self.z = np.meshgrid(xpoints, ypoints, zpoints, indexing="ij")
        self.X = np.vstack((self.x.ravel(), self.y.ravel(), self.z.ravel())).T 
        
    def _grid_spacing(self, bounds: Sequence[float], n: int) -> float:
        return (bounds[1] - bounds[0]) / n
    
    def _points(self, bounds: Sequence[float], n: int) -> np.ndarray:
        """ Creates 1D array of points in one dimension."""
        return np.linspace(bounds[0], bounds[1], n)
    
# Setup the 3D mesh.
xbounds, ybounds, zbounds = [-30, 30], [-30, 30], [0, 50]
nx = 100
mesh = Grid3D(xbounds, ybounds, zbounds, *[nx]*3)

sys = Lorenz63()
trajectory = sys.force(mesh.X)

# Plot the vector field.     
x, z = mesh.X[:, 0], mesh.X[:, 2]
U, W = trajectory[:, 0], trajectory[:, 2]

magnitude = np.sqrt(U**2 + W**2)
magnitude[magnitude == 0] = 1  # Prevent division by zero
U /= magnitude
W /= magnitude

plt.figure(figsize=(10, 7))
q_int = 6
plt.quiver(x[::q_int], z[::q_int], U[::q_int], W[::q_int], magnitude[::q_int], 
           cmap="coolwarm", scale=30, scale_units="inches", alpha=0.7)

plt.show()