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

    def __add__(self, other): 
        result = Tensor(self.data + other.data, _children=(self, other), _op='+')
        return result
    
    def __mul__(self, other):
        result = Tensor(self.data * other.data, _children=(self, other), _op='*')
        return result

if __name__ == "__main__":
    a = Tensor([2.0])
    b = Tensor([3.0])
    c = a * b
    d = c + a

    print(a)
    print(c)
    print(d)
    print(d._prev)
    print(d._op)
