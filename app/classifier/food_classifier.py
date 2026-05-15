import io
import numpy as np
from pathlib import Path
from PIL import Image


class FoodClassifier:
    """
    음식 이미지 분류기.
    TFLite 모델 파일이 있으면 TFLite로 실행, 없으면 ImageNet 기반 MobileNetV2를 자동 다운로드.
    """

    def __init__(self, model_path: str = None, labels_path: str = None, top_k: int = 5, input_size: int = 224):
        self.top_k = top_k
        self.input_size = input_size
        self.interpreter = None
        self.model = None
        self.labels: list[str] = []
        self._decode_predictions = None
        self._preprocess_input = None

        tflite_exists = model_path and Path(model_path).exists()

        if tflite_exists:
            self._load_tflite(model_path)
            if labels_path and Path(labels_path).exists():
                self._load_labels_file(labels_path)
        else:
            self._load_keras_mobilenetv2()

    # ------------------------------------------------------------------
    # 모델 로딩
    # ------------------------------------------------------------------

    def _load_tflite(self, model_path: str) -> None:
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        except ImportError:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

    def _load_keras_mobilenetv2(self) -> None:
        import tensorflow as tf
        self.model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=True)
        self._decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
        self._preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    def _load_labels_file(self, labels_path: str) -> None:
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = [line.strip() for line in f if line.strip()]

    # ------------------------------------------------------------------
    # 이미지 전처리
    # ------------------------------------------------------------------

    def _to_pil(self, image_data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_data)).convert("RGB")

    def _preprocess_for_tflite(self, image_data: bytes) -> np.ndarray:
        img = self._to_pil(image_data).resize((self.input_size, self.input_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _preprocess_for_keras(self, image_data: bytes) -> np.ndarray:
        img = self._to_pil(image_data).resize((self.input_size, self.input_size))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return self._preprocess_input(arr)

    # ------------------------------------------------------------------
    # 추론
    # ------------------------------------------------------------------

    def classify(self, image_data: bytes) -> list[dict]:
        """
        bytes 이미지를 받아 top-k 예측 결과를 반환.
        반환 형식: [{"label": str, "confidence": float}, ...]
        """
        if self.interpreter is not None:
            return self._classify_tflite(image_data)
        return self._classify_keras(image_data)

    def _classify_tflite(self, image_data: bytes) -> list[dict]:
        arr = self._preprocess_for_tflite(image_data)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # uint8 양자화 모델 처리
        if input_details[0]["dtype"] == np.uint8:
            arr = (arr * 255).astype(np.uint8)

        self.interpreter.set_tensor(input_details[0]["index"], arr)
        self.interpreter.invoke()

        probs = self.interpreter.get_tensor(output_details[0]["index"])[0]
        if probs.dtype == np.uint8:
            probs = probs.astype(np.float32) / 255.0

        top_indices = np.argsort(probs)[::-1][: self.top_k]
        return [
            {
                "label": self.labels[i] if i < len(self.labels) else f"class_{i}",
                "confidence": float(probs[i]),
            }
            for i in top_indices
        ]

    def _classify_keras(self, image_data: bytes) -> list[dict]:
        arr = self._preprocess_for_keras(image_data)
        predictions = self.model.predict(arr, verbose=0)
        decoded = self._decode_predictions(predictions, top=self.top_k)[0]
        return [
            {
                "label": label.replace("_", " ").lower(),
                "confidence": float(conf),
            }
            for _, label, conf in decoded
        ]
