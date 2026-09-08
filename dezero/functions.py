# 檔案：dezero/functions.py
from __future__ import annotations
import numpy as np
from dezero.core import Function, Variable, as_variable


class Square(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x**2

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]  # 保留 Variable，去數值化
    gx = 2 * x * gy
    return gx


class Exp(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.exp(x)

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]
    gx = exp(x) * gy  # 遞迴使用 DeZero 封裝之 exp 函式
    return gx


class Sin(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.sin(x)

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]
    gx = gy * cos(x)
    return gx


class Cos(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.cos(x)

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]
    gx = gy * -sin(x)
    return gx


class Tanh(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

  def backward(self, gy: Variable) -> Variable:
    y = self.outputs[0]()  # 取用前向輸出變數的弱引用
    gx = gy * (1.0 - y * y)
    return gx


def square(x: Variable | np.ndarray | float | int) -> Variable:
  return Square()(as_variable(x))


def exp(x: Variable | np.ndarray | float | int) -> Variable:
  return Exp()(as_variable(x))


def sin(x: Variable | np.ndarray | float | int) -> Variable:
  return Sin()(as_variable(x))


def cos(x: Variable | np.ndarray | float | int) -> Variable:
  return Cos()(as_variable(x))


def tanh(x: Variable | np.ndarray | float | int) -> Variable:
  return Tanh()(as_variable(x))

