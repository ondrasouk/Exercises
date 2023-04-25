import logging
import argparse
import sys
from pathlib import Path
from time import time
import math
import concurrent.futures
import signal

# Load should be fast when doing only "python main.py -h"
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict', action="store", type=str, default="",
                    help="Do a prediction from x_test.csv file and save output to the file.")
parser.add_argument('-s', '--save_model', action="store", type=str,
                    help="Train model and save it.")
parser.add_argument('-l', '--load_model', action="store", type=str,
                    help="Load model (skips training) and do a prediction.")
parser.add_argument('-d', '--save_data', action="store_true", default=False, help="Save x_train and y_train to npz.")
parser.add_argument('-ph', '--plot-history', action="store", type=str, default="",
                    help="Plot training history and save figures it to folder.")
parser.add_argument('--debug', action="store_true", default=False, help="More verbose logs.")
args, unknown_args = parser.parse_known_args()

# setup logging
root_logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
if args.debug:
    handler.setLevel(logging.DEBUG)
    root_logger.setLevel(logging.DEBUG)
else:
    handler.setLevel(logging.INFO)
    root_logger.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s.%(funcName)s:%(lineno)s %(message)s'))
root_logger.addHandler(handler)  # set root logger stdout output
logger = logging.getLogger(__name__)  # get logger instance for this file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tikzplotlib
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Normalization, RandomTranslation
from keras.layers import Dropout
from keras.regularizers import L1, L2, L1L2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# Set input data dtype - choose from float16, float32, float64
INPUT_DTYPE = np.float16

logger.info("Loading files")

x_train_path = Path("Dataset/Train/CSV/")
y_train_path = Path("Dataset/y_train.csv")
x_test_path = Path("Dataset/Test/CSV/")


def load_data(file: Path) -> tuple[int, np.ndarray]:
    try:
        df = pd.read_csv(file, index_col=0, sep=",", decimal=".", header=None, engine="c", low_memory=False,
                         dtype=np.float32 if INPUT_DTYPE == np.float16 else INPUT_DTYPE)
        if INPUT_DTYPE == np.float16:
            df = df.astype(INPUT_DTYPE)
    except ValueError as e:
        logger.error(f"File {file} has load error:\n{e}\nTrying to load it safely...")
        # This is something very hacky and very slow. But some files are corrupt, so...
        df = pd.read_csv(file, index_col=0, sep=",", decimal=".", header=None, engine="c",
                         dtype=str).applymap(lambda x: INPUT_DTYPE(".".join(x.split(".")[:2])))

    df.reset_index(drop=True, inplace=True)  # The index is wrong and in float...
    return int(file.name.split(".")[0].split("_")[1]), df.to_numpy()


def load_data_batch(path: Path) -> np.ndarray:
    all_files = list(path.rglob("*.csv"))
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        dfsf = executor.map(load_data, all_files, chunksize=100)
        # chunk-size is this big since there are lots of small files
    # sort by index input files so in array, depth is measurement number
    ldfs: list[tuple[int, np.ndarray]] = sorted(list(dfsf), key=lambda x: x[0])
    labels, dfs = zip(*ldfs)  # unpack
    return np.dstack(dfs)


class EarlyStoppingSIGINT(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_received = False
        self.previous_handler = None

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.previous_handler = signal.getsignal(signal.SIGINT)  # save the previous handler
        signal.signal(signal.SIGINT, self.handler)  # set the new handler

    def on_train_end(self, logs=None):
        super().on_train_end()
        signal.signal(signal.SIGINT, self.previous_handler)  # restore the previous handler

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.model.stop_training |= self.signal_received  # set the flag

    def handler(self, num: int, frame):
        logger.info("Received SIGINT, safely exiting...")  # When SIGINT (Ctrl+C) is received stop training
        self.signal_received = True


if args.save_data:
    x_train = load_data_batch(x_train_path)
    y_train = pd.read_csv(y_train_path, index_col="id").to_numpy()
    x_test = load_data_batch(x_test_path)
    logger.info("Loaded from CSV files")
    np.savez_compressed("x_train.npz", x_train)
    np.savez_compressed("y_train.npz", y_train)
    np.savez_compressed("x_test.npz", x_test)
    logger.info("Saving to numpy pickle files")
else:
    try:
        x_train = np.load("x_train.npz")["arr_0"]
        y_train = np.load("y_train.npz")["arr_0"]
        x_test = np.load("x_test.npz")["arr_0"]
        logger.info("Loaded numpy pickle files")
    except OSError:
        logger.info("Saved dataset not found. Loading it again.")
        x_train = load_data_batch(x_train_path)
        y_train = pd.read_csv(y_train_path, index_col="id").to_numpy()
        x_test = load_data_batch(x_test_path)
        logger.info("Loaded from CSV files")


n_of_classes = len(np.unique(y_train))
logger.info(f"Number of classes: {n_of_classes}")

# Normalize data
x_max = max(x_train.max(), x_test.max())
# x_train /= x_max
# x_test /= x_max
x_train = x_train.swapaxes(2, 0).reshape(*x_train.swapaxes(2, 0).shape, 1)
x_test = x_test.swapaxes(2, 0).reshape(*x_test.swapaxes(2, 0).shape, 1)

if args.load_model:
    model = load_model(args.load_model)
    train_history = np.load(f"{args.load_model.rsplit('.', 1)[0]}-history.npz", allow_pickle=True)["arr_0"].item()
else:
    y_train_encoded = to_categorical(y_train-1, num_classes=n_of_classes)

    logger.info("Creating model")
    model = Sequential()
    # model.add(RandomTranslation(0.05, 0.05, fill_mode="wrap", interpolation="nearest"))
    model.add(Normalization())
    model.add(Conv2D(256, (15, 15), activation="relu", data_format="channels_last", input_shape=x_train.shape[1:]))
    model.add(Conv2D(256, (9, 9), activation="relu", data_format="channels_last"))
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dense(n_of_classes, activation='softmax'))

    logger.info("Compiling model")
    # Create optimizer and compile model
    optimizer = Adam(learning_rate=0.00002)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['CategoricalAccuracy'],
                  jit_compile=True)
    # Print model info
    model.build(x_train.shape)
    model.summary()

    logger.info("Starting training")
    early_stopping = EarlyStoppingSIGINT(monitor="val_loss", patience=30, start_from_epoch=10,
                                         restore_best_weights=True)
    history = model.fit(x=x_train, y=y_train_encoded, validation_split=0.01, epochs=20000, batch_size=128,
                        shuffle=True, callbacks=[early_stopping])
    np.savez_compressed(f"{args.save_model.rsplit('.', 1)[0]}-history.npz", history.history)
    train_history = history.history

# Print model info
model.summary()

if args.save_model:
    logger.info("Saving model")
    model.save(args.save_model)

if args.predict:
    logger.info("Doing prediction")
    prediction = model.predict(x_test, batch_size=128)
    pred = prediction.argmax(axis=1).astype(np.uint8)+1
    pred_df = pd.DataFrame(pred, columns=["target"])
    pred_df.to_csv(args.predict, index=True, index_label="id", quoting=2)

if args.plot_history:
    Path(args.plot_history).mkdir(parents=False, exist_ok=False)
    for key in train_history.keys():
        plt.figure()
        plt.semilogy(train_history[key])
        plt.xlabel("Epochs")
        plt.ylabel(key)
        tikzplotlib.save(f"{args.plot_history}/{key}.tex")
        plt.show()
