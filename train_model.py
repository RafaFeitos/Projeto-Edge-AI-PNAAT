import os
import tensorflow as tf
from tensorflow import keras

# Suprimir os logs de informações/avisos do TensorFlow para uma saída de CI mais limpa:
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Redimensionar para (amostras, altura, largura, canais), necessário para Conv2D:
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = keras.Sequential([
        # Bloco 1 — extração de features de baixo nível:
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),

        # Bloco 2 — extração de features de médio nível:
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        # Classificador:
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia final no conjunto de teste: {accuracy:.4f}")

    return history

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_model()
    train_model(model, x_train, y_train, x_test, y_test)
    model.save("model.h5")
    print("Modelo salvo em model.h5")