# 檔案：notes/stage03_high_order_differentiation/step36.py
from pathlib import Path
import numpy as np
import pytest
from dezero import Variable
from dezero.utils import plot_dot_graph


def test_double_backprop_on_derived_gradient():
  x_val = 2.0
  x = Variable(np.array(x_val), name="x")
  y = x**2
  y.name = "y"

  # 第一次求導，必須開啟建圖旗標
  y.backward(create_graph=True)
  gx = x.grad
  gx.name = "gx"
  np.testing.assert_allclose(gx.data, 2.0 * x_val)

  # 清空輸入變數的累積梯度，避免影響第二次反向傳播
  x.cleargrad()

  # 使用梯度變數參與運算構建新目標 z
  z = gx**3 + y
  z.name = "z"

  # 對複合目標 z 再次求導
  z.backward()

  # 驗證 dz/dx 解析解: 24 * (x^2) + 2 * x = 24 * 4 + 4 = 100
  expected_dz_dx = 24.0 * (x_val**2) + 2.0 * x_val
  np.testing.assert_allclose(x.grad.data, expected_dz_dx)


def test_manual_visual_inspection_double_backprop():
  # 本地手動檢驗工作流：輸出包含導數運算的計算圖
  x = Variable(np.array(2.0), name="x")
  y = x**2
  y.backward(create_graph=True)
  gx = x.grad

  z = gx**3 + y
  z.backward()

  file_path = Path("double_backprop_graph.png")
  plot_dot_graph(z, verbose=False, to_file=str(file_path))
  assert file_path.is_file()

