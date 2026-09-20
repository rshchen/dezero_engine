# 檔案：dezero/functions.py
from __future__ import annotations

from typing import Sequence
import numpy as np
from dezero.core import Function, Variable, as_variable, Config
from dezero.utils import col2im, im2col, reshape_sum_backward, sum_to_array, as_array
import dezero.cuda as cuda

class Square(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x**2

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]  # 保留 Variable，去數值化
    gx = 2 * x * gy
    return gx

def square(x: Variable | np.ndarray | float | int) -> Variable:
  return Square()(as_variable(x))


class Exp(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.exp(x)

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]
    gx = exp(x) * gy  # 遞迴使用 DeZero 封裝之 exp 函式
    return gx

def exp(x: Variable | np.ndarray | float | int) -> Variable:
  return Exp()(as_variable(x))

class Sin(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    xp = cuda.get_array_module(x)
    return xp.sin(x)

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]
    gx = gy * cos(x)
    return gx

def sin(x: Variable | np.ndarray | float | int) -> Variable:
  return Sin()(as_variable(x))

class Cos(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.cos(x)

  def backward(self, gy: Variable) -> Variable:
    x = self.inputs[0]
    gx = gy * -sin(x)
    return gx

def cos(x: Variable | np.ndarray | float | int) -> Variable:
  return Cos()(as_variable(x))

class Tanh(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

  def backward(self, gy: Variable) -> Variable:
    y = self.outputs[0]()  # 取用前向輸出變數的弱引用
    gx = gy * (1.0 - y * y)
    return gx

def tanh(x: Variable | np.ndarray | float | int) -> Variable:
  return Tanh()(as_variable(x))


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

def transpose(
    x: Variable | np.ndarray, axes: Sequence[int] | None = None
) -> Variable:
  x = as_variable(x)
  if axes is not None:
    axes = tuple(axes)
  return Transpose(axes)(x)



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

def sum(
    x: Variable | np.ndarray,
    axis: int | Sequence[int] | None = None,
    keepdims: bool = False,
) -> Variable:
  x = as_variable(x)
  if axis is not None and not isinstance(axis, int):
    axis = tuple(axis)
  return Sum(axis, keepdims)(x)


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

def broadcast_to(
    x: Variable | np.ndarray, shape: Sequence[int] | int
) -> Variable:
  x = as_variable(x)
  target_shape = (shape,) if isinstance(shape, int) else tuple(shape)
  if x.shape == target_shape:
    return x
  return BroadcastTo(target_shape)(x)


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

def sum_to(x: Variable | np.ndarray, shape: Sequence[int] | int) -> Variable:
  x = as_variable(x)
  target_shape = (shape,) if isinstance(shape, int) else tuple(shape)
  if x.shape == target_shape:
    return x
  return SumTo(target_shape)(x)


class MatMul(Function):

  def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
    y = x.dot(W)
    return y

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW

def matmul(x: Variable | np.ndarray, W: Variable | np.ndarray) -> Variable:
  return MatMul()(as_variable(x), as_variable(W))

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

def mean_squared_error(
    x0: Variable | np.ndarray, x1: Variable | np.ndarray
) -> Variable:
  return MeanSquaredError()(as_variable(x0), as_variable(x1))

class Sigmoid(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    y = 1.0 / (1.0 + np.exp(-x))
    return y

  def backward(self, gy: Variable) -> Variable:
    y = self.outputs[0]()
    gx = gy * y * (1.0 - y)
    return gx


def sigmoid(x: Variable | np.ndarray) -> Variable:
  return Sigmoid()(as_variable(x))

class GetItem(Function):

  def __init__(self, slices):
    self.slices = slices

  def forward(self, x):
    return x[self.slices]

  def backward(self, gy):
    x, = self.inputs
    return GetItemGrad(self.slices, x.shape)(gy)

def get_item(x, slices):
  return GetItem(slices)(x)

class GetItemGrad(Function):

  def __init__(self, slices, in_shape):
    self.slices = slices
    self.in_shape = in_shape

  def forward(self, gy):
    gx = np.zeros(self.in_shape, dtype=gy.dtype)
    np.add.at(gx, self.slices, gy)
    return gx

  def backward(self, ggx):
    return get_item(ggx, self.slices)

class Softmax(Function):

  def __init__(self, axis=1):
    self.axis = axis

  def forward(self, x):
    # 減去維度最大值避免指數溢位
    x_max = x.max(axis=self.axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    y = exp_x / exp_x.sum(axis=self.axis, keepdims=True)
    return y

  def backward(self, gy):
    y = self.outputs[0]()
    # Softmax 反向傳播推導公式：gy * y - y * sum(gy * y)
    gx = gy * y
    sum_gx = gx.sum(axis=self.axis, keepdims=True)
    gx -= y * sum_gx
    return gx

def softmax(x, axis=1):
  return Softmax(axis)(x)

class SoftmaxCrossEntropy(Function):

  def forward(self, x, t):
    N = x.shape[0]

    # 1. 數值穩定的 Softmax 前向計算
    x_max = x.max(axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    y = exp_x / exp_x.sum(axis=1, keepdims=True)

    # 裁剪極端數值避免 log(0)
    eps = 1e-15
    y_clipped = np.clip(y, eps, 1.0)

    # 2. 交叉熵計算（支援索引標籤與 One-hot 標籤）
    if t.ndim == 1:
      log_p = np.log(y_clipped[np.arange(N), t])
    else:
      log_p = np.log(y_clipped) * t
    loss = -log_p.sum() / N

    # 保存預測機率供反向傳播直接使用
    self.y = y
    self.t = t
    return loss

  def backward(self, gy):
    N, CLS = self.y.shape
    gx = self.y.copy()

    # 3. 梯度計算：(y - t) / N * gy
    if self.t.ndim == 1:
      gx[np.arange(N), self.t] -= 1.0
    else:
      gx -= self.t

    gx *= gy.data / N
    return as_variable(gx), None

def softmax_cross_entropy(x, t):
  return SoftmaxCrossEntropy()(x, t)

class ReLU(Function):

  def forward(self, x):
    y = np.maximum(x, 0.0)
    return y

  def backward(self, gy):
    x, = self.inputs
    mask = x.data > 0
    gx = gy * mask
    return gx

def relu(x):
  return ReLU()(x)



class Dropout(Function):

  def __init__(self, dropout_ratio):
    self.dropout_ratio = dropout_ratio
    self.mask = None

  def forward(self, x):
    # 生成保留機率為 (1 - dropout_ratio) 的二元遮罩
    scale = 1.0 - self.dropout_ratio
    mask = np.random.rand(*x.shape) > self.dropout_ratio
    self.mask = mask
    y = x * mask / scale
    return y

  def backward(self, gy):
    scale = 1.0 - self.dropout_ratio
    gx = gy * self.mask / scale
    return gx


def dropout(x, dropout_ratio=0.5):
  x = as_variable(x)
  if Config.train:
    return Dropout(dropout_ratio)(x)
  else:
    return x


class Conv2d(Function):

  def __init__(self, stride=1, pad=0):
    super().__init__()
    self.stride = stride
    self.pad = pad

  def forward(self, x, W, b=None):
    KH, KW = W.shape[2:]
    col = im2col(x, (KH, KW), self.stride, self.pad, to_matrix=False)

    N, C, KH, KW, OH, OW = col.shape
    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * OH * OW, C * KH * KW)
    W_row = W.reshape(W.shape[0], -1).T

    t = np.dot(col, W_row)
    if b is not None:
      t += b
    t = t.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)
    return t

  def backward(self, gy):
    has_bias = len(self.inputs) == 3
    if has_bias:
      x, W, b = self.inputs
    else:
      x, W = self.inputs
    
    KH, KW = W.shape[2:]



    # 2. 計算針對權重的梯度
    col = im2col(x.data, (KH, KW), self.stride, self.pad, to_matrix=False)
    N, C, KH, KW, OH, OW = col.shape
    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * OH * OW, C * KH * KW)

    gy_t = gy.data.transpose(0, 2, 3, 1).reshape(-1, gy.shape[1])
    gW = np.dot(col.T, gy_t).transpose(1, 0).reshape(W.shape)
    gW = as_variable(gW)

    # 3. 計算針對輸入的梯度
    W_row = W.data.reshape(W.shape[0], -1).T
    gcol = np.dot(gy_t, W_row.T)
    gx = col2im(gcol, x.shape, (KH, KW), self.stride, self.pad, to_matrix=True)
    gx = as_variable(gx)

    if not has_bias:
        return gx, gW

    # 1. 計算針對偏誤的梯度
    gb = gy.sum(axis=(0, 2, 3))
    return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
  if b is None:
    return Conv2d(stride, pad)(x, W)
  return Conv2d(stride, pad)(x, W, b)

# 檔案：dezero/functions.py
class Pooling(Function):

  def __init__(self, kernel_size, stride=1, pad=0):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad

  def forward(self, x):
    col = im2col(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
    N, C, KH, KW, OH, OW = col.shape
    col = col.transpose(0, 1, 4, 5, 2, 3).reshape(N, C, OH, OW, -1)

    self.arg_max = col.argmax(axis=-1)
    y = col.max(axis=-1)
    return y

  def backward(self, gy):
    x = self.inputs[0]
    KH, KW = (
        (self.kernel_size, self.kernel_size)
        if isinstance(self.kernel_size, int)
        else self.kernel_size
    )

    N, C, OH, OW = gy.shape
    gcol = np.zeros((N, C, OH, OW, KH * KW), dtype=gy.dtype)
    gcol[
        np.arange(N).reshape(N, 1, 1, 1),
        np.arange(C).reshape(1, C, 1, 1),
        np.arange(OH).reshape(1, 1, OH, 1),
        np.arange(OW).reshape(1, 1, 1, OW),
        self.arg_max,
    ] = gy.data

    gcol = gcol.reshape(N, C, OH, OW, KH, KW).transpose(0, 1, 4, 5, 2, 3)
    gx = col2im(
        gcol, x.shape, self.kernel_size, self.stride, self.pad, to_matrix=False
    )
    return as_variable(gx)


def pooling(x, kernel_size, stride=1, pad=0):
  return Pooling(kernel_size, stride, pad)(x)


def linear(
    x: Variable | np.ndarray,
    W: Variable | np.ndarray,
    b: Variable | np.ndarray | None = None,
) -> Variable:
  t = x @ W
  if b is None:
    return t

  y = t + b
  t.data = None  # 手動釋放中間張量數值，減少記憶體佔用
  return y




def accuracy(y, t):
  y = as_variable(y)
  t = as_variable(t)

  pred = y.data.argmax(axis=1)
  pred = pred.reshape(t.shape)
  result = (pred == t.data)
  acc = np.mean(result)
  return Variable(as_array(acc))














