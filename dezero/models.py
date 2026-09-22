# 檔案：dezero/models.py
from __future__ import annotations
from dezero.layers import Layer, Linear
from dezero.utils import get_file, plot_dot_graph
from collections.abc import Callable
import dezero.functions as F
import dezero.layers as L
import numpy as np

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



class VGG16(Model):
  WEIGHTS_PATH = "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"

  def __init__(self, pretrained=False):
    super().__init__()
    # Block 1
    self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
    # Block 2
    self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
    # Block 3
    self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
    self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
    self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
    # Block 4
    self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
    # Block 5
    self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
    self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
    # Classifier
    self.fc6 = L.Linear(4096)
    self.fc7 = L.Linear(4096)
    self.fc8 = L.Linear(1000)

    if pretrained:
      weights_path = get_file(self.WEIGHTS_PATH)
      self.load_weights(weights_path)

  def forward(self, x):
    # Block 1
    x = F.relu(self.conv1_1(x))
    x = F.relu(self.conv1_2(x))
    x = F.pooling(x, 2, 2)
    # Block 2
    x = F.relu(self.conv2_1(x))
    x = F.relu(self.conv2_2(x))
    x = F.pooling(x, 2, 2)
    # Block 3
    x = F.relu(self.conv3_1(x))
    x = F.relu(self.conv3_2(x))
    x = F.relu(self.conv3_3(x))
    x = F.pooling(x, 2, 2)
    # Block 4
    x = F.relu(self.conv4_1(x))
    x = F.relu(self.conv4_2(x))
    x = F.relu(self.conv4_3(x))
    x = F.pooling(x, 2, 2)
    # Block 5
    x = F.relu(self.conv5_1(x))
    x = F.relu(self.conv5_2(x))
    x = F.relu(self.conv5_3(x))
    x = F.pooling(x, 2, 2)
    # Classifier
    x = F.reshape(x, (x.shape[0], -1))
    x = F.dropout(F.relu(self.fc6(x)))
    x = F.dropout(F.relu(self.fc7(x)))
    x = self.fc8(x)
    return x

  @staticmethod
  def preprocess(image, size=(224, 224), dtype=np.float32):
    image = image.convert("RGB")
    if size:
      image = image.resize(size)
    image = np.asarray(image, dtype=dtype)
    # RGB 轉 BGR
    image = image[:, :, ::-1]
    # 扣除 ImageNet 均值
    image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
    # 軸向調整為 (C, H, W)
    image = image.transpose((2, 0, 1))
    # 增加 Batch 維度 -> (1, C, H, W)
    return image[np.newaxis, ...]

