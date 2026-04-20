import os
import tensorflow as tf

# Reduz logs do TensorFlow no CI
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_model(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo carregado: {model_path}")
    return model


def convert_to_tflite(model: tf.keras.Model, output_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Reduz tamanho do modelo para Edge AI:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"Modelo TFLite salvo em {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    model = load_model("model.h5")
    convert_to_tflite(model, "model.tflite")