import numpy as np

class Tensor: 
    def __init__(self, data, _children=(), _op=''):
        
        self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)

        self._backward = lambda: None

        self._prev = set(_children)

        self._op = _op
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"