import os
import tensorflow as tf

# Reduz logs do TensorFlow no CI
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_model(model_path: str) -> tf.keras.Model:
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo carregado: {model_path}")
    return model


def convert_dynamic_range(model: tf.keras.Model, output_path: str) -> int:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic Range Quantization — converte pesos float32 para int8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return len(tflite_model)


def convert_float16(model: tf.keras.Model, output_path: str) -> int:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Float16 Quantization — converte pesos float32 para float16
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    return len(tflite_model)


if __name__ == "__main__":
    model = load_model("model.h5")

    h5_size = os.path.getsize("model.h5") / 1024

    dr_size = convert_dynamic_range(model, "model.tflite") / 1024
    f16_size = convert_float16(model, "model_float16.tflite") / 1024

    print("\n--- Comparativo de Otimização ---")
    print(f"model.h5          : {h5_size:.1f} KB  (referência)")
    print(f"model.tflite      : {dr_size:.1f} KB  (Dynamic Range, redução: {(1 - dr_size/h5_size)*100:.1f}%)")
    print(f"model_float16     : {f16_size:.1f} KB  (Float16, redução: {(1 - f16_size/h5_size)*100:.1f}%)")