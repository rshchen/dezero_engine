from __future__ import annotations
import numpy as np
import weakref

# 純量轉型工具函式
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
  def __init__(self, data: np.ndarray):
    if data is not None:
      # 嚴格檢查傳入資料型別是否為 np.ndarray
      if not isinstance(data, np.ndarray):
          raise TypeError(f'{type(data)} is not supported')

    self.data = data
    self.grad: np.ndarray | None = None  # 必須宣告，避免被判定為純 NoneType
    self.creator: Function | None = None 
    self.generation: int = 0

  def set_creator(self, func: 'Function'):
    self.creator = func
    self.generation = func.generation + 1 # 輸出變數的世代為算子世代加 1
  def cleargrad(self):
    # 重置梯度為 None
    self.grad = None


  def backward(self):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

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
      # 將 gys 解包傳入算子的 backward 運算
      gxs = f.backward(*gys)
      # 確保 gxs 為 tuple 結構
      if not isinstance(gxs, tuple):
        gxs = (gxs, )

      # 梯度累加與圖繪製追蹤
      for x, gx in zip(f.inputs, gxs):
        if x.grad is None:
          x.grad = gx
        else:
          # 必須使用 x.grad + gx，避免 in-place 修改引發記憶體參照污染
          x.grad = x.grad + gx  # 使用加法進行梯度累加
        # 若輸入變數含有 creator，將其壓入堆疊繼續向上追蹤
        if x.creator is not None:
          add_func(x.creator)


class Function:

  def __call__(self, *inputs: Variable) -> Variable | tuple[Variable, ...]:
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    # 確保 ys 為 tuple 結構
    if not isinstance(ys, tuple):
      ys = (ys,)

    # 算子世代等於輸入變數中世代最大值
    self.generation = max([x.generation for x in inputs])
    # 封裝 Variable 陣列
    outputs = [Variable(as_array(y)) for y in ys]
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

class Square(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return x**2

  def backward(self, gy: np.ndarray) -> np.ndarray:
    x = self.inputs[0].data
    gx = 2 * x * gy  # dL/dx = 2x * dL/dy
    return gx


class Exp(Function):

  def forward(self, x: np.ndarray) -> np.ndarray:
    return np.exp(x)

  def backward(self, gy: np.ndarray) -> np.ndarray:
    x = self.inputs[0].data
    gx = np.exp(x) * gy # dL/dx = exp(x) * dL/dy
    return gx
  

# 算子快捷函式封裝
def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


class Add(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return x0 + x1
  def backward(self, gy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
  return Add()(x0, x1)