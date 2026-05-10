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

        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward

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

        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        result._backward = _backward

        return result

    def __neg__(self):
        """
        Negates the tensor element-wise. 
        
        Used internally by __sub__ and anywhere a sign flip is needed.
        """
        result = Tensor(-self.data, _children=(self,), _op='neg')
        
        def _backward():
            self.grad += -result.grad
        result._backward = _backward
        
        return result

    def __pow__(self, exponent):
        """
        Raises the tensor to a scalar exponent element-wise. 
        
        Used by __truediv__ (via exponent=-1) and directly in 
        loss functions such as MSE (via exponent=2). 
        """
        result = Tensor(self.data ** exponent, _children=(self,), _op='**')
        
        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * result.grad
        result._backward = _backward

        return result

    def __radd__(self, other):
        """
        Handles addition when the Tensor is the right-hand opperand.

        Called by Python when a non-Tensor type appears on the left
        side of +. Delegates to __add__ so the computation graph is 
        built identically regardless of operand order.
        """
        return self + other

    def __rmul__(self, other):
        """
        Handles multiplication when the Tensor is the right-hand
        operand. 

        Called by Python when a non-Tensor type appears on the left
        side of *. Delegates to __mul__ so the computation graph is
        built identically regardless of operand order.
        """
        return self * other

    def __sub__(self, other):
        """
        Computes element-wise subtraction of two tensors. 

        Implemented as addition of the negation so gradients flow 
        correctly through the existing __add__ and __neg__ backward
        closures. Scalars are automatically wrapped in a Tensor.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other):
        """
        Computes element-wise division of two tensors. 

        Implemented as multiplication by the reciprocal – self * (other ** -1)
        so gradients flow correctly through the existing __mul__ and
        __pow__ backward closures. Scalars are auomatically wrapped 
        in a Tensor.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def exp(self):
        """
        Computes the element-wise natural exponential of the tensor. 

        The backward closure exploits the unique property of the 
        exponential function, its derivative is itself, so the 
        local gradient is simply the forward output value of result.data.
        """
        result = Tensor(np.exp(self.data), _children=(self,), _op='exp')
        
        def _backward():
            self.grad += result.data * result.grad
        result._backward = _backward

        return result

    def log(self):
        """
        Computes the element-wise natural logarithm of the tensor.
        
        The local gradient is 1/x, so each element of self.grad is 
        scaled by the reciprocal of the corresponding input value. 
        Note: undefined for non-positive values — inputs must be 
        strictly positive.
        """
        result = Tensor(np.log(self.data), _children=(self,), _op='log')

        def _backward():
            self.grad += 1/self.data * result.grad
        result._backward = _backward

        return result

    def relu(self):
        """
        Applies the Rectified Linear Unit activation function 
        element-wise, clamping negative values to zero and passing
        positive values through unchanged.

        The backward closure uses a binary mask — gradients pass
        through where the input was positive and are blocked where
        it was not, reflecting the piecewise derivative of ReLU.
        """
        result = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')
        
        def _backward():
            self.grad += (result.data > 0) * result.grad
        result._backward = _backward

        return result

    def sum(self):
        """
        Reduces the tensor to a scalar by summing all elements.
        
        The backward closure fans the upstream gradient back out
        equally to every element of self, since each element 
        contributed to the sum with a coefficient of 1.
        """
        result = Tensor(np.sum(self.data), _children=(self,), _op='sum')
        
        def _backward():
            self.grad += result.grad * np.ones_like(self.data)
        result._backward = _backward

        return result

    def backward(self):
        """
        Applies topological sort to neurons, initializes output gradient to 1,
        and propagates to child neurons to obtain their gradients using
        chain rule.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


