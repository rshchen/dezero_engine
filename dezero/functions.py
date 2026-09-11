# 檔案：dezero/functions.py
from __future__ import annotations

from typing import Sequence
import numpy as np
from dezero.core import Function, Variable, as_variable
from dezero.utils import reshape_sum_backward, sum_to_array

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

class Sum(Function):

  def __init__(
      self,
      axis: int | tuple[int, ...] | None = None,
      keepdims: bool = False,
  ) -> None:
    self.axis = axis
    self.keepdims = keepdims

  def forward(self, x: np.ndarray) -> np.ndarray:
    self.x_shape = x.shape
    y = np.sum(x, axis=self.axis, keepdims=self.keepdims)
    return y

  def backward(self, gy: Variable) -> Variable:
    # 1. 補齊被壓縮的軸（長度設為 1），使秩與輸入對齊
    gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    # 2. 沿求和軸複製填充回原始形狀
    gx = broadcast_to(gy, self.x_shape)
    return gx

class BroadcastTo(Function):

  def __init__(self, shape: tuple[int, ...]) -> None:
    self.shape = shape

  def forward(self, x: np.ndarray) -> np.ndarray:
    self.x_shape = x.shape
    y = np.broadcast_to(x, self.shape)
    return y

  def backward(self, gy: Variable) -> Variable:
    gx = sum_to(gy, self.x_shape)
    return gx

class SumTo(Function):

  def __init__(self, shape: tuple[int, ...]) -> None:
    self.shape = shape

  def forward(self, x: np.ndarray) -> np.ndarray:
    self.x_shape = x.shape
    y = sum_to_array(x, self.shape)
    return y

  def backward(self, gy: Variable) -> Variable:
    gx = broadcast_to(gy, self.x_shape)
    return gx

class MatMul(Function):

  def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
    y = x.dot(W)
    return y

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW

class MeanSquaredError(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    diff = x0 - x1
    y = (diff**2).sum() / len(x0)
    return y

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    x0, x1 = self.inputs
    diff = x0 - x1
    gx0 = gy * (2.0 / len(x0)) * diff
    gx1 = -gx0
    return gx0, gx1







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


def sum(
    x: Variable | np.ndarray,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Variable:
  x = as_variable(x)
  if axis is not None and not isinstance(axis, int):
    axis = tuple(axis)
  return Sum(axis, keepdims)(x)

def broadcast_to(
    x: Variable | np.ndarray, shape: Sequence[int] | int
) -> Variable:
  x = as_variable(x)
  target_shape = (shape,) if isinstance(shape, int) else tuple(shape)
  if x.shape == target_shape:
    return x
  return BroadcastTo(target_shape)(x)

def sum_to(x: Variable | np.ndarray, shape: Sequence[int] | int) -> Variable:
  x = as_variable(x)
  target_shape = (shape,) if isinstance(shape, int) else tuple(shape)
  if x.shape == target_shape:
    return x
  return SumTo(target_shape)(x)

def matmul(x: Variable | np.ndarray, W: Variable | np.ndarray) -> Variable:
  return MatMul()(as_variable(x), as_variable(W))

def mean_squared_error(
    x0: Variable | np.ndarray, x1: Variable | np.ndarray
) -> Variable:
  return MeanSquaredError()(as_variable(x0), as_variable(x1))