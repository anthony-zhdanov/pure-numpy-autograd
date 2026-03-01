import numpy as np

class Tensor:
    """
    Represents a node within a multi-layer simple neural network.
    """ 
    def __init__(self, data, _children=(), _op=''):
        
        self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)

        self._backward = lambda: None

        self._prev = set(_children)

        self._op = _op
    
    def __repr__(self):
        """
        Representation dunder method allowing a Tensor itself to be an output.
        """
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Addition dunder method allowing sum of Tensors' values to be calculated. To be used as the underlying function used for any instance of Tensor addition.
        """
        other = other if isinstance(other, Tensor) else Tensor(other) 
        result = Tensor(self.data + other.data, _children=(self, other), _op='+')
        return result
    
    def __mul__(self, other):
        """
        Multiplication dunder method allowing Tensors' products to be calculated. Arguably the most important function within neural net architecture.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data, _children=(self, other), _op='*')
        return result

    def __neg__(self):
        """
        Negation dunder method to be used to negate a Tensor instance. To be used within cross-entropy loss calculations.
        """
        result = Tensor(-self.data, _children=(self,), _op='neg')
        return result

    def __pow__(self, exponent):
        result = Tensor(self.data ** exponent, _children=(self,), _op='**')
        return result

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def exp(self):
        result = Tensor(np.exp(self.data), _children=(self,), _op='exp')
        return result

    def log(self):
        result = Tensor(np.log(self.data), _children=(self,), _op='log')
        return result

    def relu(self):
        result = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')
        return result

    def sum(self):
        result = Tensor(np.sum(self.data), _children=(self,), _op='sum')
        return result

