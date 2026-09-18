# 檔案：dezero/layers.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from dezero.core import Parameter
import dezero.functions as F
import weakref
import dezero.cuda as cuda

class Layer:

  def __init__(self):
    self._params = set()

  def __setattr__(self, name: str, value: object):
    
    if isinstance(value, (Parameter, Layer)):
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
        obj = self.__dict__[name]
        if isinstance(obj, Layer):
          # 透過 yield from 將子層所持有的參數逐一向外轉發
          yield from obj.params()
        else:
          # 當前物件為具體的 Parameter 實體，直接釋出
          yield obj

  def cleargrads(self):
    # 批次清空所有已註冊參數的梯度
    for param in self.params():
      param.cleargrad()

  def to_cpu(self):
    for param in self.params():
      param.to_cpu()

  def to_gpu(self):
    for param in self.params():
      param.to_gpu()

  def _flatten_params(self, params_dict, parent_key=""):
    """遞迴展平層級參數，將結果就地寫入傳入的 params_dict。"""
    for name in self._params:
      obj = self.__dict__[name]
      key = f"{parent_key}/{name}" if parent_key else name

      if isinstance(obj, Layer):
        obj._flatten_params(params_dict, parent_key=key)
      else:
        params_dict[key] = obj

  def save_weights(self, path):
    """將所有層級參數序列化為 .npz 檔案。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params_dict = {}
    self._flatten_params(params_dict)

    array_dict = {}
    for key, param in params_dict.items():
      if param is not None and param.data is not None:
        # 若資料位於 GPU，轉換至 CPU 陣列以確保通用儲存相容性
        array_dict[key] = cuda.as_numpy(param.data)

    # 透過 ** 字典解包，以具名引數方式傳遞給 np.savez_compressed
    np.savez_compressed(path, **array_dict)

  def load_weights(self, path):
    """從 .npz 檔案反序列化並注入參數陣列。"""
    path = Path(path)
    if not path.exists():
      raise FileNotFoundError(f"Weight file not found: {path}")

    params_dict = {}
    self._flatten_params(params_dict)

    with np.load(path) as npz_file:
      for key, param in params_dict.items():
        if key in npz_file and param is not None:
          param_data = npz_file[key]
          # 確保還原時符合目標層目前所在的硬體裝置 (NumPy / CuPy)
          param.data = cuda.get_array_module(param).asarray(param_data)

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

