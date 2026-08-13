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
    self.grad: np.ndarray | None = None  # 必須宣告，避免被判定為純 NoneType
    self.creator: 'Function' | None = None # Function 還沒有定義 所以要用字串表示

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

  def __call__(self, *inputs: Variable) -> Variable | tuple[Variable, ...]:
    xs = [x.data for x in inputs]
    ys = self.forward(*xs)
    # 確保 ys 為 tuple 結構
    if not isinstance(ys, tuple):
      ys = (ys,)
    # 封裝 Variable 陣列
    self.outputs = [Variable(as_array(y)) for y in ys]
    # 建立血緣關係：將輸出變數的 creator 指向自身
    for output in self.outputs:
      output.set_creator(self)
    self.inputs = inputs  # 保存輸入變數，供 backward 計算使用
    
    return self.outputs[0] if len(self.outputs)==1 else tuple(self.outputs)

  def forward(self, *xs: np.ndarray) -> np.ndarray | tuple[np.ndarray, ...]:
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


class Add(Function):

  def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    return x0 + x1


def add(x0: Variable, x1: Variable) -> Variable:
  return Add()(x0, x1)