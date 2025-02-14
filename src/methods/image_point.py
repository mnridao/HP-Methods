from typing import Callable, Optional
import numpy as np

class ImagePointAdvector:
    """ 
    Image point method, with or without nudging."""

    def __init__(self, x_ref: np.ndarray, force_ref: Callable[[np.ndarray], np.ndarray], 
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
        self.force_ref = force_ref
        
        # Nudging parameters.
        self.eta = eta
        self.nudge = nudge 
        
    def force(self, y: np.ndarray) -> np.ndarray:
        """ 
        Calculates the average force over the neighbourhood of y."""
        
        # Find the N nearest neighbours to y.
        x_neighbours = self._nearest_neighbours(y)
        
        # Average force in the neighbourhood.
        f_avg = np.sum(self.force_ref(x_neighbours), axis=0)/self.N
        
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