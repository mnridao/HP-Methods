from typing import Callable
import numpy as np

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