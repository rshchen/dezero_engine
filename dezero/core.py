import numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data

class Function:

  def __call__(self, input: Variable) -> Variable:
    x = input.data
    y = self.forward(x)
    output = Variable(y)
    return output

  def forward(self, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

class Square(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x**2


class Exp(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.exp(x)
  

