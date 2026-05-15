"""
TFLite 모델 다운로드 스크립트.
models/ 폴더에 MobileNetV2 quantized TFLite 모델과 labels.txt를 저장합니다.

사용법:
    python scripts/download_model.py
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

# TensorFlow 공식 MobileNetV2 quantized TFLite (ImageNet 1001 classes)
MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v2_1.0_224_quant_and_labels.zip"
MODEL_FILENAME = "mobilenet_v2_1.0_224_quant.tflite"
LABELS_FILENAME = "labels_mobilenet_quant_v1_224.txt"


def download_and_extract() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    tflite_path = MODEL_DIR / "mobilenet_v2_1.0_224.tflite"
    labels_path = MODEL_DIR / "labels.txt"

    if tflite_path.exists() and labels_path.exists():
        print("모델 파일이 이미 존재합니다.")
        return

    print(f"모델 다운로드 중: {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL) as response:
        zip_bytes = response.read()

    print("압축 해제 중...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            print(f"  포함 파일: {name}")

        # TFLite 모델 저장
        tflite_candidates = [n for n in zf.namelist() if n.endswith(".tflite")]
        if tflite_candidates:
            data = zf.read(tflite_candidates[0])
            tflite_path.write_bytes(data)
            print(f"저장됨: {tflite_path}")

        # Labels 저장
        label_candidates = [n for n in zf.namelist() if n.endswith(".txt")]
        if label_candidates:
            data = zf.read(label_candidates[0])
            labels_path.write_bytes(data)
            print(f"저장됨: {labels_path}")

    print("완료! .env의 MODEL_PATH / LABELS_PATH를 확인하세요.")


if __name__ == "__main__":
    try:
        download_and_extract()
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
