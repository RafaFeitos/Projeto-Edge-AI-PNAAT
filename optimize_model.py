import tensorflow as tf
import os

# Suprimir logs do TensorFlow para saída limpa no CI
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_model(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo carregado: {model_path}")
    return model


def convert_to_tflite(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Conversão para TFLite concluída.")
    return tflite_model


if __name__ == "__main__":
    model = load_model("model.h5")
    convert_to_tflite(model)