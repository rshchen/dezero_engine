# 檔案：notes/stage05_deep_learning_applications/step58.py
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from dezero import Variable
from dezero.models import VGG16
from dezero.utils import get_file


def test_get_file_cache_path(tmp_path, monkeypatch):
  # 將 Path.home() 暫時置換為 pytest 的暫存目錄
  monkeypatch.setattr(Path, "home", lambda: tmp_path)

  # 模擬本機檔案路徑回傳行為
  dummy_file = tmp_path / ".dezero" / "test.txt"
  dummy_file.parent.mkdir(parents=True, exist_ok=True)
  dummy_file.write_text("ok")

  file_path = get_file("http://example.com/test.txt")
  assert Path(file_path).name == "test.txt"
  assert Path(file_path).exists()


def test_vgg16_preprocess_shape():
  # 建立一張隨機 300x400 的 RGB 影像
  dummy_img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8))
  x = VGG16.preprocess(dummy_img)

  # 驗證輸出形狀符合批次化四維標準 (N, C, H, W)
  assert x.shape == (1, 3, 224, 224)
  assert x.dtype == np.float32


def test_vgg16_forward_shape():
  # 建立無預訓練權重的骨架以進行形狀驗證
  model = VGG16(pretrained=False)
  x = Variable(np.random.randn(1, 3, 224, 224).astype(np.float32))

  y = model(x)

  # 最終全連接層應輸出 1000 個類別的機率分佈 logits
  assert y.shape == (1, 1000)


@pytest.mark.slow
def test_vgg16_inference_zebra():
  url = "https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg"
  img_path = get_file(url)
  img = Image.open(img_path)

  x = VGG16.preprocess(img)
  model = VGG16(pretrained=True)
  y = model(x)

  predict_id = np.argmax(y.data)
  # ImageNet zebra 類別編號
  assert predict_id == 340

