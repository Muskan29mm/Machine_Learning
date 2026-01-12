"""
cnn_implementation.py

Simple, self-contained Convolutional Neural Network (CNN) example using
TensorFlow / Keras. Supports MNIST and CIFAR-10, with data augmentation,
training, evaluation, and model saving.

Usage examples:
  python cnn_implementation.py --dataset cifar10 --epochs 20 --batch-size 64 --train
  python cnn_implementation.py --dataset mnist --epochs 10 --augment --train
  Add --save to persist model and training plots (requires --train)

Dependencies:
  pip install tensorflow matplotlib numpy

"""
import argparse
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except Exception as e:
    print("TensorFlow import failed:", e)
    print("Install TensorFlow first: pip install tensorflow")
    sys.exit(1)


def get_dataset(name="cifar10"):
    """Load and preprocess dataset. Returns (x_train, y_train), (x_test, y_test), input_shape, num_classes"""
    name = name.lower()
    if name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        num_classes = 10
    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # expand channels
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset: {}".format(name))

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    input_shape = x_train.shape[1:]

    # Flatten labels to 1D for sparse categorical crossentropy
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    return (x_train, y_train), (x_test, y_test), input_shape, num_classes


def build_cnn(input_shape, num_classes, dropout_rate=0.3):
    """Create a simple CNN model.

    Architecture:
      [Conv -> Conv -> Pool] x2 -> Flatten -> Dense -> Output
    """
    model = models.Sequential()

    # Input layer: accept images with shape `input_shape`
    model.add(layers.Input(shape=input_shape))

    # Hidden layers (convolutional feature extractors)
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))

    # Second convolutional block (hidden)
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))

    # Transition to dense layers
    model.add(layers.Flatten())

    # Hidden dense layer(s)
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout_rate))

    # Output layer: softmax over `num_classes`
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def plot_history(history, savepath=None):
    """Plot training/validation accuracy and loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history.get("val_loss", []), label="val loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history.get("val_accuracy", []), label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
    else:
        plt.show()


def train_and_evaluate(dataset="cifar10", epochs=20, batch_size=64, augment=False, save_dir="models", lr=1e-3, train=False, save=False):
    # If not training, avoid loading datasets or creating directories.
    if not train:
        # Use default input shapes for common datasets so we can build and
        # display the model summary without downloading or loading data.
        ds = dataset.lower()
        if ds == "cifar10":
            input_shape = (32, 32, 3)
            num_classes = 10
        elif ds == "mnist":
            input_shape = (28, 28, 1)
            num_classes = 10
        else:
            raise ValueError("Unsupported dataset for summary-only mode: {}".format(dataset))

        model = build_cnn(input_shape, num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.summary()
        print("Model built and summarized. Exiting without training or saving.")
        return model, None

    # Training path: load actual dataset
    (x_train, y_train), (x_test, y_test), input_shape, num_classes = get_dataset(dataset)

    model = build_cnn(input_shape, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "best_model.h5")

    cb = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    if save:
        cb.insert(0, callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1))

    if augment:
        print("Using data augmentation")
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )
        datagen.fit(x_train)

        steps_per_epoch = math.ceil(len(x_train) / batch_size)
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(x_test, y_test),
            callbacks=cb,
            verbose=2,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=cb,
            verbose=2,
        )

    # Evaluate
    results = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss, Test accuracy:", results)

    if save:
        final_path = os.path.join(save_dir, "final_model.h5")
        model.save(final_path)
        print("Model saved to:", final_path)
        plot_history(history, savepath=os.path.join(save_dir, "training_plot.png"))
    else:
        print("Training complete. Model not saved (use --save to enable saving).")

    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN (Keras) on MNIST or CIFAR-10")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"], help="dataset to use")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augment", action="store_true", help="use data augmentation")
    parser.add_argument("--save-dir", default="models", help="directory to save models and plots")
    parser.add_argument("--train", action="store_true", help="actually train the model (default: show summary only)")
    parser.add_argument("--save", action="store_true", help="save model and training plots (requires --train)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Enforce summary-only mode: ignore --train and --save flags and exit
    print("Summary-only mode enforced: building model and printing summary only (no download, no training, no saving).")
    train_and_evaluate(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=False,
        save_dir=args.save_dir,
        lr=args.lr,
        train=False,
        save=False,
    )
