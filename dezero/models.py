# 檔案：dezero/models.py
from __future__ import annotations
from dezero.layers import Layer, Linear
from dezero.utils import plot_dot_graph
from collections.abc import Callable
import dezero.functions as F

class Model(Layer):

  def plot(self, *inputs, to_file: str = "model.png"):
    y = self.forward(*inputs)
    return plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):

  def __init__(
      self,
      fc_output_sizes: tuple[int, ...] | list[int],
      activation: Callable = F.sigmoid,
  ):
    super().__init__()
    self.activation = activation
    self.layers = []

    # 動態宣告並註冊各線性層
    for i, out_size in enumerate(fc_output_sizes):
      layer = Linear(out_size)
      setattr(self, f"l{i}", layer)  # 觸發 __setattr__，自動將子層註冊進 self._params
      self.layers.append(layer)

  def forward(self, x):
    # 前向傳播：除最後一層外，其餘層皆串接激活函數
    for layer in self.layers[:-1]:
      x = self.activation(layer(x))
    return self.layers[-1](x)