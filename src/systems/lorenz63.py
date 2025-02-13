from dataclasses import dataclass
import numpy as np

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