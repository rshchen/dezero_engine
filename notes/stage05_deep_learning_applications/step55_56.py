# 檔案：notes/stage05_deep_learning_applications/step56.py
import pytest
from dezero.utils import get_conv_outsize


def test_get_conv_outsize_same():
  # 輸入 32，濾鏡 3，填充 1，步幅 1 -> 維持 32
  out_size = get_conv_outsize(input_size=32, filter_size=3, stride=1, pad=1)
  assert out_size == 32


def test_get_conv_outsize_strided():
  # 輸入 32，濾鏡 4，填充 1，步幅 2 -> 減半為 16
  out_size = get_conv_outsize(input_size=32, filter_size=4, stride=2, pad=1)
  assert out_size == 16


def test_get_conv_outsize_floor_division():
  # 輸入 10，濾鏡 3，填充 0，步幅 2 -> (10 + 0 - 3) // 2 + 1 = 4
  out_size = get_conv_outsize(input_size=10, filter_size=3, stride=2, pad=0)
  assert out_size == 4

