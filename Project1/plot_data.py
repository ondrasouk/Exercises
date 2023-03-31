import matplotlib.pyplot as plt
import pandas as pd
import logging
import sys
import argparse
from itertools import combinations

# setup logging
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action="store_true", default=False)
args, unknown_args = parser.parse_known_args()

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


# Load files
excluded_keys = ["m_power", "Tosc", "Tmix"]

x_train = pd.read_csv("x_train.csv", index_col=0, usecols=lambda x: x not in excluded_keys)
y_train = pd.read_csv("y_train.csv", index_col="id")
x_test = pd.read_csv("x_test.csv", index_col=0, usecols=lambda x: x not in excluded_keys)

y_types = range(y_train.target.min(), y_train.target.max()+1)
color_dict = {k: f"C{n}" for k, n in zip(y_types, range(len(y_types)))}
x_train_colors = [color_dict[t] for t in y_train.target]
for x, y in combinations(x_train.keys(), 2):
    logger.info(f"plotting {x}, {y}")
    x_train.plot.scatter(x=x, y=y, c=x_train_colors)
    plt.show()
