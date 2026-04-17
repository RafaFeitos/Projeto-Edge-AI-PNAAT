import os
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers

# Suprimir os logs de informações/avisos do TensorFlow para uma saída de CI mais limpa:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_and_preprocess_data():
    """Carregamento (reshape) e aplicação da normalização do dataset MNIST"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Redimensionar para (amostras, altura, largura, canais), necessário para Conv2D:
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print(f"Train samples : {x_train.shape[0]}")
    print(f"Test  samples : {x_test.shape[0]}")
    print(f"Input shape   : {x_train.shape[1:]}")