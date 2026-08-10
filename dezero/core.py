from typing import Optional

import numpy as np

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
    self.grad: Optional[np.ndarray] = None  # 必須宣告，避免被判定為純 NoneType
    self.creator: Optional['Function'] = None # Function 還沒有定義 所以要用字串表示

  def set_creator(self, func: 'Function'):
    self.creator = func

  def backward(self):
    if self.grad is None:
      self.grad = np.ones_like(self.data)

    funcs = []
    if self.creator is not None:
      funcs.append(self.creator)

    while funcs:
      # 從堆疊中取出當前待處理的算子
      f = funcs.pop()
      # 存取該算子的輸入與輸出變數
      x, y = f.input, f.output
      # 呼叫算子的 backward 計算輸入變數梯度
      x.grad = f.backward(y.grad)

      # 若輸入變數含有 creator，將其壓入堆疊繼續向上追蹤
      if x.creator is not None:
        funcs.append(x.creator)
        

class Function:

  def __call__(self, input: Variable) -> Variable:
    x = input.data
    y = self.forward(x)
    # 確保前向計算結果轉換為 np.ndarray 後才封裝為 Variable
    output = Variable(as_array(y))
    # 建立血緣關係：將輸出變數的 creator 指向自身
    output.set_creator(self)
    self.input = input  # 保存輸入變數，供 backward 計算使用
    self.output = output
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
  

# 算子快捷函式封裝
def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)