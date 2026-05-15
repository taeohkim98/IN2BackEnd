"""
공통 픽스처.
test_images/ 폴더에 이미지를 넣으면 실제 파일로 테스트하고,
없으면 PIL로 합성한 더미 이미지로 대체합니다.
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"


def _make_dummy_image(width: int = 224, height: int = 224) -> bytes:
    """랜덤 RGB 더미 이미지를 JPEG bytes로 반환."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def dummy_image_bytes() -> bytes:
    return _make_dummy_image()


@pytest.fixture
def real_image_bytes() -> bytes:
    """test_images/ 첫 번째 이미지 사용. 없으면 더미 이미지."""
    images = sorted(
        list(TEST_IMAGES_DIR.glob("*.jpg"))
        + list(TEST_IMAGES_DIR.glob("*.jpeg"))
        + list(TEST_IMAGES_DIR.glob("*.png"))
    )
    if images:
        return images[0].read_bytes()
    return _make_dummy_image()


@pytest.fixture
def all_test_images() -> list[tuple[str, bytes]]:
    """test_images/ 폴더의 모든 이미지를 (파일명, bytes) 리스트로 반환."""
    images = sorted(
        list(TEST_IMAGES_DIR.glob("*.jpg"))
        + list(TEST_IMAGES_DIR.glob("*.jpeg"))
        + list(TEST_IMAGES_DIR.glob("*.png"))
    )
    return [(p.name, p.read_bytes()) for p in images]
