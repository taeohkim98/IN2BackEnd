"""
FoodClassifier 단위 테스트.
실제 TF 모델을 로드하므로 처음 실행 시 다운로드 시간이 걸릴 수 있습니다.
"""
import pytest

from app.classifier.food_classifier import FoodClassifier


@pytest.fixture(scope="module")
def classifier() -> FoodClassifier:
    return FoodClassifier(top_k=5, input_size=224)


class TestFoodClassifierInit:
    def test_loads_without_tflite(self):
        clf = FoodClassifier()
        # TFLite 파일 없으면 Keras 모델 사용
        assert clf.model is not None or clf.interpreter is not None

    def test_keras_fallback_when_no_model_file(self):
        clf = FoodClassifier(model_path="nonexistent_model.tflite")
        assert clf.model is not None


class TestClassify:
    def test_returns_list(self, classifier, dummy_image_bytes):
        results = classifier.classify(dummy_image_bytes)
        assert isinstance(results, list)

    def test_top_k_results(self, classifier, dummy_image_bytes):
        results = classifier.classify(dummy_image_bytes)
        assert len(results) <= classifier.top_k

    def test_result_schema(self, classifier, dummy_image_bytes):
        results = classifier.classify(dummy_image_bytes)
        for r in results:
            assert "label" in r
            assert "confidence" in r
            assert isinstance(r["label"], str)
            assert 0.0 <= r["confidence"] <= 1.0

    def test_confidence_sorted_descending(self, classifier, dummy_image_bytes):
        results = classifier.classify(dummy_image_bytes)
        confs = [r["confidence"] for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_real_image_if_available(self, classifier, real_image_bytes):
        """test_images/ 폴더에 실제 음식 사진이 있을 때만 실행."""
        results = classifier.classify(real_image_bytes)
        assert len(results) > 0
        print("\n[실제 이미지 예측 결과]")
        for r in results:
            print(f"  {r['label']}: {r['confidence']:.4f}")

    def test_all_test_images(self, classifier, all_test_images):
        """test_images/ 폴더의 모든 이미지를 순서대로 분류."""
        if not all_test_images:
            pytest.skip("test_images/ 폴더에 이미지 없음")

        for name, data in all_test_images:
            results = classifier.classify(data)
            assert len(results) > 0, f"{name}: 결과 없음"
            top = results[0]
            print(f"\n[{name}] -> {top['label']} ({top['confidence']:.4f})")
