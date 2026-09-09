# 檔案：dezero/functions.py
from __future__ import annotations
import numpy as np
from dezero.core import Function, Variable, as_variable
from typing import Sequence


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


class Reshape(Function):

  def __init__(self, shape: tuple[int, ...]) -> None:
    self.shape = shape
    self.x_shape: tuple[int, ...] | None = None

  def forward(self, x: np.ndarray) -> np.ndarray:
    self.x_shape = x.shape
    y = x.reshape(self.shape)
    return y

  def backward(self, gy: Variable) -> Variable:
    return reshape(gy, self.x_shape)

def square(x: Variable | np.ndarray | float | int) -> Variable:
  return Square()(as_variable(x))


class Transpose(Function):

  def __init__(self, axes: tuple[int, ...] | None = None) -> None:
    self.axes = axes

  def forward(self, x: np.ndarray) -> np.ndarray:
    y = np.transpose(x, self.axes)
    return y

  def backward(self, gy: Variable) -> Variable:
    if self.axes is None:
      return transpose(gy)

    # 計算通用逆置換：若 axes = (1, 2, 0)，則 inv_axes = (2, 0, 1)
    inv_axes = tuple(np.argsort(self.axes))
    return transpose(gy, inv_axes)


def exp(x: Variable | np.ndarray | float | int) -> Variable:
  return Exp()(as_variable(x))


def sin(x: Variable | np.ndarray | float | int) -> Variable:
  return Sin()(as_variable(x))


def cos(x: Variable | np.ndarray | float | int) -> Variable:
  return Cos()(as_variable(x))


def tanh(x: Variable | np.ndarray | float | int) -> Variable:
  return Tanh()(as_variable(x))

def reshape(
    x: Variable | np.ndarray, shape: int | Sequence[int]
) -> Variable:
  x = as_variable(x)
  if isinstance(shape, int):
    target_shape = (shape,)
  else:
    target_shape = tuple(shape)

  if x.shape == target_shape:
    return as_variable(x)
  return Reshape(target_shape)(x)

def transpose(
    x: Variable | np.ndarray, axes: Sequence[int] | None = None
) -> Variable:
  x = as_variable(x)
  if axes is not None:
    axes = tuple(axes)
  return Transpose(axes)(x)
