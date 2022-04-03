"""

This is a POC for a moving live-cusum graph

"""

import collections
import time
from datetime import datetime, timedelta
from pprint import pprint
from turtle import color

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib.animation import FuncAnimation

from utils.ftx_rest import ftx as FTX

plt.style.use("seaborn")
COLOR = "white"
mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR


class Watcher(FTX):
    def __init__(self, ticker, cperc):
        super().__init__()
        self.ticker = ticker
        self.cperc = cperc

    def pull(self):
        """
        pulls txs for the past x seconds
        """
        now = int(datetime.now().timestamp() * 1000)
        # then = now - 1000 * 60 * 60 * 4
        then = now - 1000 * 60 * 50
        points = pd.DataFrame(
            [x["info"] for x in self.fetch_trades(self.ticker, since=then, until=now)]
        )
        points["price"] = points["price"].astype(float)
        return points

    def detect_hits(self):
        """
        adds columns to detect cusum hits, returning mapping of hit times to prices
        """

        self.tick_cache["cusum_coord"] = self.tick_cache[["time", "price"]].to_dict("records")

        self.tick_cache["diff"] = (
            self.tick_cache["price"] / self.tick_cache["price"].shift(1) - 1.0
        ) * 100
        self.tick_cache.dropna(inplace=True)

        self.tick_cache["diff_pos"] = np.where(
            self.tick_cache["diff"] > 0.0, self.tick_cache["diff"], 0.0
        )
        self.tick_cache["diff_neg"] = np.where(
            self.tick_cache["diff"] < 0.0, self.tick_cache["diff"] * -1, 0.0
        )

        for side in ["pos", "neg"]:
            self.tick_cache["cusum_{}".format(side)] = self.tick_cache[
                "diff_{}".format(side)
            ].cumsum()
            self.tick_cache["cusum_{}".format(side)] = (
                self.tick_cache["cusum_{}".format(side)] % self.cperc
            )
            self.tick_cache["{}_bound_hit".format(side)] = np.where(
                self.tick_cache["cusum_{}".format(side)]
                < self.tick_cache["cusum_{}".format(side)].shift(1),
                "HIT",
                "-",
            )

        self.tick_cache["hit"] = np.where(
            (self.tick_cache["pos_bound_hit"] == "HIT")
            | (self.tick_cache["neg_bound_hit"] == "HIT"),
            self.tick_cache["cusum_coord"],
            "no_hit",
        )

        hits = pd.DataFrame([x for x in self.tick_cache["hit"] if x != "no_hit"])
        hits["abs_perc_change"] = abs((hits["price"] / hits["price"].shift(1) - 1) * 100)
        hits.dropna(inplace=True)

        return hits

    def refresh(self, i):

        self.tick_cache = self.pull()
        self.hit_cache = self.detect_hits()

        self.refresh_cache()

        all_price = collections.deque(self.tick_cache["price"])
        hit_price = collections.deque(self.hit_cache["price"])

        all_time = collections.deque(self.tick_cache["time"])
        hit_time = collections.deque(self.hit_cache["time"])

        self.ax.cla()
        self.ax.set_title(self.ticker)

        self.ax.plot(all_time, all_price, color="#00FF7B", zorder=0, linewidth=1)
        self.ax.scatter(hit_time, hit_price, color="red", zorder=10)

        for x in range(len(hit_price)):
            self.ax.hlines(
                hit_price[x],
                xmin=min(all_time),
                xmax=hit_time[x],
                color="red",
                linestyles="dashed",
            )

        plt.xticks(hit_time, labels=[format_time(x) for x in hit_time], rotation=45, ha="right")
        plt.yticks(hit_price)

    def run(self):
        """initialises the plot which will be blank"""

        self.fig = plt.figure(figsize=(12, 6), facecolor="#000000")

        self.ax = plt.subplot()
        self.ax.grid(color="grey")
        self.ax.set_yscale("log")
        self.ax.set_facecolor("#000000")

        # animate
        ani = FuncAnimation(self.fig, self.refresh, interval=5 * 1000)
        plt.show()


def format_time(dt):
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f+00:00")
    return datetime.strftime(dt, "%H:%M:%S")


@logger.catch
def main():

    btc_watch = Watcher("ETH/USDT", cperc=0.75)
    btc_watch.run()

    # while 1:
    #     time.sleep(5)
    #     btc_watch.refresh_plot()


if __name__ == "__main__":
    main()
