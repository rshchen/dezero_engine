# 檔案：dezero/core.py
from __future__ import annotations

import contextlib
from typing import Sequence
import weakref
import numpy as np
from dezero.utils import as_array


# 轉型成變數的工具函式
def as_variable(obj: Variable | np.ndarray | float | int) -> Variable:
  if isinstance(obj, Variable):
    return obj
  return Variable(as_array(obj))

# 定義全域組態開關類別
class Config:
  enable_backprop: bool = True


# 實作基於 contextmanager 的通用組態切換器
@contextlib.contextmanager
def using_config(name: str, value: bool):
  old_value = getattr(Config, name)
  setattr(Config, name, value)
  try:
    yield
  finally:
    setattr(Config, name, old_value)


def no_grad():
  return using_config("enable_backprop", False)

class Variable:
  def __init__(self, data: np.ndarray, name: str | None = None):
    if data is not None:
      # 嚴格檢查傳入資料型別是否為 np.ndarray
      if not isinstance(data, np.ndarray):
          raise TypeError(f'{type(data)} is not supported')

    self.data = data
    self.name = name
    self.grad: Variable | None = None  # 必須宣告，避免被判定為純 NoneType
    self.creator: Function | None = None 
    self.generation: int = 0

  # 使用屬性取值器代理 ndarray 屬性
  @property
  def shape(self) -> tuple[int, ...]:
    return self.data.shape

  @property
  def ndim(self) -> int:
    return self.data.ndim

  @property
  def size(self) -> int:
    return self.data.size

  @property
  def dtype(self) -> np.dtype:
    return self.data.dtype

  @property
  def T(self) -> Variable:
    import dezero.functions as F
    return F.transpose(self)

  # 實作長度協定魔術方法，對接全域 len()
  def __len__(self) -> int:
    return len(self.data)

  # 實作字串顯示魔術方法
  def __repr__(self) -> str:
    if self.data is None:
      return "variable(None)"
    p = str(self.data).replace("\n", "\n" + " " * 9)
    return f"variable({p})"

  def set_creator(self, func: Function):
    self.creator = func
    self.generation = func.generation + 1 # 輸出變數的世代為算子世代加 1
  def cleargrad(self):
    # 重置梯度為 None
    self.grad = None

  def reshape(self, *shape: int | Sequence[int]) -> Variable:
    import dezero.functions as F
    if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
      target_shape = shape[0]
    else:
      target_shape = shape
    return F.reshape(self, target_shape)

  def transpose(self, *axes: int | Sequence[int]) -> Variable:
    import dezero.functions as F
    if len(axes) == 0:
      target_axes = None
    elif len(axes) == 1 and isinstance(axes[0], (list, tuple)):
      target_axes = axes[0]
    elif len(axes) == 1 and axes[0] is None:
      target_axes = None
    else:
      target_axes = axes
    return F.transpose(self, target_axes)

  def sum(
      self,
      axis: int | Sequence[int] | None = None,
      keepdims: bool = False,
  ) -> Variable:
    import dezero.functions as F
    return F.sum(self, axis=axis, keepdims=keepdims)

  def broadcast_to(self, shape: Sequence[int] | int) -> Variable:
    import dezero.functions as F

    return F.broadcast_to(self, shape=shape)


  def sum_to(self, shape: Sequence[int] | int) -> Variable:
    import dezero.functions as F

    return F.sum_to(self, shape=shape)

  def backward(self, retain_grad: bool = False, create_graph: bool = False):
    if self.grad is None:
      self.grad = Variable(np.ones_like(self.data))

    funcs: list[Function] = []
    seen_set: set[Function] = set()

    def add_func(f: Function):
      # 避免算子重複加入並保持世代排序
      if f not in seen_set:
        funcs.append(f)
        seen_set.add(f)
        # 依 generation 遞增排序，確保 pop() 能取出最大世代算子
        funcs.sort(key=lambda x: x.generation)

    if self.creator is not None:
      add_func(self.creator)

    while funcs:
      # 從堆疊中取出當前待處理的算子
      f = funcs.pop()
      # 收集所有輸出變數的梯度
      gys = [output().grad for output in f.outputs]
      # 根據 create_graph 決定是否在反向傳播時保留導函數計算圖
      with using_config("enable_backprop", create_graph):
        # 將 gys 解包傳入算子的 backward 運算
        gxs = f.backward(*gys)
        # 確保 gxs 為 tuple 結構
        if not isinstance(gxs, tuple):
          gxs = (gxs, )

        # 梯度累加
        for x, gx in zip(f.inputs, gxs):
          if x.grad is None:
            x.grad = gx
          else:
            # 必須使用 x.grad + gx，避免 in-place 修改引發記憶體參照污染
            x.grad = x.grad + gx  # 透過 Variable 重載的加法建立圖節點
          # 若輸入變數含有 creator，將其壓入堆疊繼續向上追蹤
          if x.creator is not None:
            add_func(x.creator)

      # 中間梯度即時釋放：若不保留中間梯度，走訪完算子後立即將其 outputs 的 grad 歸零
      if not retain_grad:
        for y in f.outputs:
          y().grad = None



class Function:

  def __call__(self, *inputs: Variable) -> Variable | tuple[Variable, ...]:
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    # 確保 ys 為 tuple 結構
    if not isinstance(ys, tuple):
      ys = (ys,)


    # 封裝 Variable 陣列
    outputs = [Variable(as_array(y)) for y in ys]
    # 僅在啟用反向傳播模式時，才建構計算圖血緣
    if Config.enable_backprop:
      # 算子世代等於輸入變數中世代最大值
      self.generation = max([x.generation for x in inputs])
      # 建立血緣關係：將輸出變數的 creator 指向自身
      for output in outputs:
        output.set_creator(self)
      self.inputs = inputs  # 保存輸入變數，供 backward 計算使用
      self.outputs = [weakref.ref(output) for output in outputs] # 將輸出變數包裝為 weakref 弱引用，避免循環參照

    
    return outputs[0] if len(outputs)==1 else tuple(outputs)

  def forward(self, *xs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
    raise NotImplementedError()

  def backward(self, *gys: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
    raise NotImplementedError()


class Parameter(Variable):
  pass


class Add(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0_shape = x0.shape
    self.x1_shape = x1.shape
    y = x0 + x1
    return y

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    gx0 = gy
    gx1 = gy
    if self.x0_shape != self.x1_shape:
      import dezero.functions as F  # 延遲匯入以避免循環依賴

      gx0 = F.sum_to(gx0, self.x0_shape)
      gx1 = F.sum_to(gx1, self.x1_shape)
    return gx0, gx1


class Mul(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    y = x0 * x1
    return y

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    x0, x1 = self.inputs
    gx0 = gy * x1
    gx1 = gy * x0
    if x0.shape != x1.shape:
      import dezero.functions as F  # 延遲匯入以避免循環依賴

      gx0 = F.sum_to(gx0, x0.shape)
      gx1 = F.sum_to(gx1, x1.shape)
    return gx0, gx1


class Neg(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return -x

  def backward(self, gy: np.ndarray) -> np.ndarray:
    return -gy


class Sub(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0_shape = x0.shape
    self.x1_shape = x1.shape
    return x0 - x1

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    gx0 = gy
    gx1 = -gy
    if self.x0_shape != self.x1_shape:
      import dezero.functions as F

      gx0 = F.sum_to(gx0, self.x0_shape)
      gx1 = F.sum_to(gx1, self.x1_shape)
    return gx0, gx1

class Div(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    self.x0_shape = x0.shape
    self.x1_shape = x1.shape
    return x0 / x1

  def backward(self, gy: Variable) -> tuple[Variable, Variable]:
    x0, x1 = self.inputs
    gx0 = gy / x1
    gx1 = gy * (-x0 / (x1**2))
    if self.x0_shape != self.x1_shape:
      import dezero.functions as F

      gx0 = F.sum_to(gx0, self.x0_shape)
      gx1 = F.sum_to(gx1, self.x1_shape)
    return gx0, gx1



class Pow(Function):

  def __init__(self, c: int | float):
    self.c = c

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x**self.c

  def backward(self, gy: np.ndarray) -> np.ndarray:
    x = self.inputs[0]
    c = self.c
    return c * (x ** (c - 1)) * gy


# 封裝算子輔助函式
def add(x0: Variable, x1: Variable | float | int) -> Variable:
  x1 = as_variable(x1)
  return Add()(x0, x1)


def mul(x0: Variable, x1: Variable | float | int) -> Variable:
  x1 = as_variable(x1)
  return Mul()(x0, x1)


def neg(x: Variable) -> Variable:
  return Neg()(x)


def sub(x0: Variable, x1: Variable | float | int) -> Variable:
  x1 = as_variable(x1)
  return Sub()(x0, x1)


def rsub(x0: Variable, x1: Variable | float | int) -> Variable:
  x1 = as_variable(x1)
  return Sub()(x1, x0)


def div(x0: Variable, x1: Variable | float | int) -> Variable:
  x1 = as_variable(x1)
  return Div()(x0, x1)


def rdiv(x0: Variable, x1: Variable | float | int) -> Variable:
  x1 = as_variable(x1)
  return Div()(x1, x0)


def pow(x: Variable, c: int | float) -> Variable:
  return Pow(c)(x)

def matmul(x: Variable | np.ndarray, W: Variable | np.ndarray) -> Variable:
  import dezero.functions as F

  return F.matmul(x, W)


def rmatmul(W: Variable | np.ndarray, x: Variable | np.ndarray) -> Variable:
  import dezero.functions as F

  return F.matmul(x, W)

# 動態掛載魔術方法
def setup_variable():
  Variable.__add__ = add
  Variable.__radd__ = add  # 加法滿足交換律之右側運算子
  Variable.__mul__ = mul
  Variable.__rmul__ = mul
  Variable.__neg__ = neg  # 一元負號魔術方法
  Variable.__sub__ = sub
  Variable.__rsub__ = rsub
  Variable.__truediv__ = div
  Variable.__rtruediv__ = rdiv
  Variable.__pow__ = pow

  # 矩陣乘法 @ 運算子多載
  Variable.matmul = matmul
  Variable.__matmul__ = matmul
  Variable.__rmatmul__ = rmatmul
