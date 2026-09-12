# 檔案：dezero/layers.py
from __future__ import annotations
import numpy as np
from dezero.core import Parameter
import dezero.functions as F
import weakref

class Layer:

  def __init__(self):
    self._params = set()

  def __setattr__(self, name: str, value: object):
    # 若指派的屬性為 Parameter，則自動加入參數註冊集合
    if isinstance(value, Parameter):
      self._params.add(name)
    # 委託 object 原生機制安全寫入實例字典，避免無窮遞迴
    super().__setattr__(name, value)

  def __call__(self, *inputs):
    outputs = self.forward(*inputs)
    if not isinstance(outputs, tuple):
      outputs = (outputs,)

    # 使用 weakref 保留輸入輸出資訊供查驗，不干擾記憶體釋放週期
    self.inputs = [weakref.ref(x) for x in inputs]
    self.outputs = [weakref.ref(y) for y in outputs]

    return outputs if len(outputs) > 1 else outputs[0]

  def forward(self, *inputs):
    raise NotImplementedError()

  def params(self):
    # 透過生成器惰性走訪當前層所屬的所有可學習參數
    for name in self._params:
      yield self.__dict__[name]

  def cleargrads(self):
    # 批次清空所有已註冊參數的梯度
    for param in self.params():
      param.cleargrad()



class Linear(Layer):

  def __init__(
      self,
      out_size: int,
      nobias: bool = False,
      dtype: np.dtype = np.float32,
      in_size: int | None = None,
  ):
    super().__init__()
    self.in_size = in_size
    self.out_size = out_size
    self.dtype = dtype

    self.W = Parameter(None, name="W")
    # 如果没有指定 in_size，則延後處理
    if self.in_size is not None:
      self._init_W()

    if nobias:
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

  def _init_W(self):
    I, O = self.in_size, self.out_size
    # LeCun 初始化
    W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1.0 / I)
    self.W.data = W_data

  def forward(self, x):
    # 延遲初始化：首次前向傳播動態推斷輸入形狀
    if self.W.data is None:
      self.in_size = x.shape[1]
      self._init_W()

    y = F.linear(x, self.W, self.b)
    return y

