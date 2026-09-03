import numpy as np


def test_package_import_and_operators():
  # 1. 驗證從套件頂層直接導入核心元件
  from dezero import Variable

  x = Variable(np.array(2.0))
  # 驗證 setup_variable 是否在套件載入時成功執行（支援運算子多載）
  y = 3.0 * x + 4.0 - 2.0 / x

  # y = 3*2 + 4 - 2/2 = 6 + 4 - 1 = 9.0
  np.testing.assert_allclose(y.data, 9.0)

  y.backward()
  # dy/dx = 3 + 0 - (-2 / x^2) = 3 + 2/(4) = 3.5
  np.testing.assert_allclose(x.grad, 3.5)


def test_top_level_specialized_functions():
  # 2. 驗證頂層匯出的特化算子
  from dezero import Variable, exp, square

  x = Variable(np.array(2.0))
  y = square(x) + exp(x)

  # y = 4.0 + e^2
  expected = 4.0 + np.exp(2.0)
  np.testing.assert_allclose(y.data, expected)

  y.backward()
  # dy/dx = 2x + e^x = 4 + e^2
  expected_grad = 4.0 + np.exp(2.0)
  np.testing.assert_allclose(x.grad, expected_grad)


def test_top_level_no_grad():
  # 3. 驗證頂層匯出的 no_grad 上下文管理器
  from dezero import Variable, no_grad

  x = Variable(np.array(3.0))
  with no_grad():
    y = x * 2.0 + 1.0

  # no_grad 模式下計算圖血緣中斷
  assert y.creator is None

