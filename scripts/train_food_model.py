"""
Food-101 데이터셋으로 MobileNetV2 fine-tuning 후 TFLite 변환.

사용법:
    pip install tensorflow-datasets
    python scripts/train_food_model.py

소요 시간: CPU 약 1-2시간 / GPU 약 10-20분
결과물: models/mobilenet_v2_1.0_224.tflite, models/labels.txt
"""
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 101


def preprocess(sample):
    image = tf.image.resize(sample["image"], (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    label = tf.one_hot(sample["label"], NUM_CLASSES)
    return image, label


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def save_labels(ds_info):
    label_names = ds_info.features["label"].names
    labels_path = MODEL_DIR / "labels.txt"
    labels_path.write_text("\n".join(label_names))
    print(f"라벨 저장: {labels_path} ({len(label_names)}개)")
    return label_names


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = MODEL_DIR / "mobilenet_v2_1.0_224.tflite"
    tflite_path.write_bytes(tflite_model)
    print(f"TFLite 저장: {tflite_path} ({len(tflite_model) // 1024} KB)")


def main():
    print("Food-101 데이터셋 다운로드 중... (첫 실행 시 약 5GB)")
    (ds_train, ds_val), ds_info = tfds.load(
        "food101",
        split=["train", "validation"],
        with_info=True,
        as_supervised=False,
    )

    save_labels(ds_info)

    train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                    .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print("모델 학습 시작...")
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train, validation_data=val, epochs=EPOCHS)

    print("TFLite 변환 중...")
    convert_to_tflite(model)
    print("\n완료! 이제 pytest로 테스트하세요.")


if __name__ == "__main__":
    main()
