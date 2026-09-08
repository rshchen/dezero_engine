# 檔案：notes/stage03_high_order_differentiation/step35.py
from pathlib import Path
import numpy as np
import pytest
from dezero import Variable, cos, exp, sin, square, tanh
from dezero.utils import plot_dot_graph


def test_tanh_higher_order_derivatives():
  x_val = 1.0
  x = Variable(np.array(x_val), name="x")
  y = tanh(x)
  y.name = "y"

  # 一階求導
  y.backward(create_graph=True)
  gx = x.grad
  gx.name = "gx"
  expected_y = np.tanh(x_val)
  expected_gx = 1.0 - expected_y**2
  np.testing.assert_allclose(gx.data, expected_gx)

  # 二階求導
  x.cleargrad()
  gx.backward(create_graph=True)
  g2x = x.grad
  g2x.name = "g2x"
  expected_g2x = -2.0 * expected_y * (1.0 - expected_y**2)
  np.testing.assert_allclose(g2x.data, expected_g2x)

  # 三階求導
  x.cleargrad()
  g2x.backward()
  g3x = x.grad
  expected_g3x = -2.0 * (1.0 - expected_y**2) ** 2 + 4.0 * (expected_y**2) * (
      1.0 - expected_y**2
  )
  np.testing.assert_allclose(g3x.data, expected_g3x)


def test_manual_visual_inspection_workflow():
  # 本地手動檢驗工作流：產出實體圖檔供肉眼檢視導函數計算圖
  x = Variable(np.array(1.0), name="x")
  y = tanh(x)
  y.backward(create_graph=True)
  gx = x.grad
  gx.name = "gx"

  file_path = Path("tanh_derivative_check.png")
  plot_dot_graph(gx, verbose=False, to_file=str(file_path))
  assert file_path.is_file()

