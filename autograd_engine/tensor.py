import numpy as np

class Tensor:
    """
    Represents a node in a computation graph. 
    Implemented by wrapping a numpy array.

    Tracks the data, gradient, and each individual node's
    connectivity to be used for low-level implementation 
    of backpropogation
    """ 
    def __init__(self, data, _children=(), _op=''):
        
        self.data = np.array(data, dtype=np.float64)

        self.grad = np.zeros_like(self.data)

        self._backward = lambda: None

        self._prev = set(_children)

        self._op = _op
    
    def __repr__(self):
        """
        Returns a string representation of the Tensor.
        Shows it's data and current gradient.
        """
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Computes the element-wise addition of two tensors.

        Scalars are automatically wrapped in a Tensor. 
        Records both operands as _children in the
        computation graph for the backward pass.
        """
        other = other if isinstance(other, Tensor) else Tensor(other) 
        result = Tensor(self.data + other.data, _children=(self, other), _op='+')
        return result
    
    def __mul__(self, other):
        """
        Computes element-wise multiplication of two tensors. 

        Scalars are automatically wrapped in a Tensor. 
        Records both operands as _children in the
        computation graph for the backward pass.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data, _children=(self, other), _op='*')
        return result

    def __neg__(self):
        """
        Negates the tensor element-wise. Used internally by __sub__ and anywhere a sign flip is needed.
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

