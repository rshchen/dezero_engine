from typing import Optional

import numpy as np



class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad: Optional[np.ndarray] = None  # 必須宣告，避免被判定為純 NoneType

class Function:

  def __call__(self, input: Variable) -> Variable:
    x = input.data
    y = self.forward(x)
    output = Variable(y)
    self.input = input  # 保存輸入變數，供 backward 計算使用
    return output

  def forward(self, x: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

  def backward(self, gy: np.ndarray) -> np.ndarray:
    raise NotImplementedError()

class Square(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x**2

  def backward(self, gy: np.ndarray) -> np.ndarray:
    x = self.input.data
    gx = 2 * x * gy  # dL/dx = 2x * dL/dy
    return gx


class Exp(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.exp(x)

  def backward(self, gy: np.ndarray) -> np.ndarray:
    x = self.input.data
    gx = np.exp(x) * gy # dL/dx = exp(x) * dL/dy
    return gx
  

