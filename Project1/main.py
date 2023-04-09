import logging
import argparse
import pickle
import sys
from pathlib import Path
from time import time
import math

# Load should be fast when doing only "python main.py -h"
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--predict', action="store", type=str, default="",
                    help="Do a prediction from x_test.csv file and save output to the file.")
parser.add_argument('-s', '--save_model', action="store", type=str,
                    help="Train model and save it.")
parser.add_argument('-l', '--load_model', action="store", type=str,
                    help="Load model (skips training) and do a prediction.")
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
import cv2
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.regularizers import L1, L2, L1L2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

logger.info("Loading files")
excluded_keys = ["m_power", "Tosc", "Tmix"]  # These were excluded in the task description.
# (With them, the categorical accuracy can go up to 99.97% to almost 100%

x_train = pd.read_csv("x_train.csv", index_col=0, usecols=lambda x: x not in excluded_keys)
y_train = pd.read_csv("y_train.csv", index_col="id")
x_test = pd.read_csv("x_test.csv", index_col=0, usecols=lambda x: x not in excluded_keys)

logger.info("Checking data")
# Check, that columns are same.
assert len(x_train.columns) == len(x_test.columns)
assert all(x_train.columns == x_test.columns)
assert len(x_train) == len(y_train)

n_of_classes = int(y_train.nunique())


if args.load_model:
    model = load_model(args.load_model)
else:
    y_train_encoded = to_categorical(y_train-1, num_classes=n_of_classes)

    logger.info("Creating model")
    model = Sequential()
    model.add(Dense(256, input_dim=x_train.shape[1], activation='sigmoid'))
    # model.add(Dropout(0.1))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(n_of_classes, activation='softmax'))

    logger.info("Compiling model")
    # Create optimizer and compile model
    optimizer = Adam(learning_rate=0.00002)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['CategoricalAccuracy'])
    # Print model info
    model.summary()

    logger.info("Starting training")
    early_stopping = EarlyStopping(monitor="val_loss", patience=500)
    history = model.fit(x=x_train.to_numpy(), y=y_train_encoded, validation_split=0.01, epochs=20000, batch_size=128,
                        shuffle=True, callbacks=[early_stopping])
    history_df = pd.DataFrame(history.history)
    if args.save_model:
        model.save(args.save_model)
        history_df.to_csv(f"{args.save_model.rsplit('.', 1)[0]}.csv")
    if args.plot_history:
        Path(args.plot_history).mkdir(parents=False, exist_ok=False)
        for key in history.history.keys():
            plt.figure()
            plt.semilogy(history.history[key])
            plt.xlabel("Epochs")
            plt.ylabel(key)
            tikzplotlib.save(f"{args.plot_history}/{key}.tex")
            plt.show()

# Print model info
model.summary()

if args.predict:
    prediction = model.predict(x_test, batch_size=128)
    pred = prediction.argmax(axis=1).astype(np.uint8)+1
    pred_df = pd.DataFrame(pred, columns=["target"])
    pred_df.to_csv(args.predict, index=True, index_label="id", quoting=2)
