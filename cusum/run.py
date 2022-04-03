"""

This is a POC for a moving live-cusum graph

"""

import collections
import time
from datetime import datetime, timedelta
from pprint import pprint
from turtle import color
from types import new_class

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

basic_labels = ["id", "price", "size", "side", "liquidation", "time"]


class Watcher(FTX):
    def __init__(self, ticker, cperc, wait_sec, tick_limit=2000):
        super().__init__()
        self.ticker = ticker
        self.cperc = cperc
        self.wait_sec = wait_sec
        self.tick_limit = tick_limit

        self.tick_cache = self.pull()
        self.hit_cache = detect_hits(self.tick_cache, self.cperc)
        time.sleep(self.wait_sec)

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

    def refresh_cache(self):

        """

        pull the new ticks and merge with the old
        only keep last 1000 ticks
        find ticks that are after the last cusum hit
        concat these to the hit cache
        delete hits that are older than the tick cache

        """

        new_ticks = self.pull()
        self.tick_cache = self.tick_cache.merge(new_ticks, how="outer", on=basic_labels)

        if len(self.tick_cache) > self.tick_limit:
            self.tick_cache = self.tick_cache.iloc[-self.tick_limit :]

        new_to_count = self.tick_cache[
            self.tick_cache["time"] >= self.hit_cache["time"].max()
        ].copy()[basic_labels]

        new_hits = detect_hits(new_to_count, self.cperc)

        if len(new_hits) > 0:
            self.hit_cache = pd.concat([self.hit_cache, new_hits])

        self.hit_cache = self.hit_cache[self.hit_cache["time"] >= self.tick_cache["time"].min()]

    def refresh(self, i):

        self.refresh_cache()

        all_price = collections.deque(self.tick_cache["price"])
        hit_price = collections.deque(self.hit_cache["price"])

        all_time = collections.deque(self.tick_cache["time"])
        hit_time = collections.deque(self.hit_cache["time"])
        vwap_ = vwap(self.tick_cache)

        self.ax.cla()
        self.ax.set_title(self.ticker)

        self.ax.plot(all_time, all_price, color="#00FF7B", zorder=5, linewidth=1)
        self.ax.plot(all_time, vwap_, color="#FC03FC", zorder=10, linewidth=1)
        self.ax.scatter(hit_time, hit_price, color="red", zorder=15)

        # for x in range(len(hit_price)):
        # self.ax.hlines(
        #     hit_price[x],
        #     xmin=min(all_time),
        #     xmax=hit_time[x],
        #     color="red",
        #     linestyles="dashed",
        # )
        # self.ax.vlines(
        #     hit_time[x],
        #     ymin=min(all_price),
        #     ymax=hit_price[x],
        #     color="red",
        #     linestyles="dashed",
        # )

        plt.xticks(hit_time, labels=[format_time(x) for x in hit_time], rotation=45, ha="right")
        plt.yticks(hit_price)

    def run(self):
        """initialises the plot which will be blank"""

        self.fig = plt.figure(figsize=(12, 6), facecolor="#000000")

        self.ax = plt.subplot()
        self.ax.grid(color="grey", zorder=0)
        self.ax.set_yscale("log")
        self.ax.set_facecolor("#000000")

        # animate
        ani = FuncAnimation(self.fig, self.refresh, interval=self.wait_sec * 1000)
        plt.show()


def format_time(dt):
    dt = datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f+00:00")
    return datetime.strftime(dt, "%H:%M:%S")


def vwap(x):
    df = x.copy()
    df["a"] = (df["price"].astype(float) * df["size"].astype(float)).cumsum()
    df["vwap"] = df["a"] / df["size"].astype(float).cumsum()
    return collections.deque(df["vwap"])


def detect_hits(df, cperc):
    """
    adds columns to detect cusum hits, returning mapping of hit times to prices
    """
    tick_df = df[basic_labels].copy()

    tick_df["cusum_coord"] = tick_df[["time", "price"]].to_dict("records")

    tick_df["diff"] = (tick_df["price"] / tick_df["price"].shift(1) - 1.0) * 100
    tick_df.dropna(inplace=True)

    tick_df["diff_pos"] = np.where(tick_df["diff"] > 0.0, tick_df["diff"], 0.0)
    tick_df["diff_neg"] = np.where(tick_df["diff"] < 0.0, tick_df["diff"] * -1, 0.0)

    for side in ["pos", "neg"]:
        tick_df["cusum_{}".format(side)] = tick_df["diff_{}".format(side)].cumsum()
        tick_df["cusum_{}".format(side)] = tick_df["cusum_{}".format(side)] % cperc
        tick_df["{}_bound_hit".format(side)] = np.where(
            tick_df["cusum_{}".format(side)] < tick_df["cusum_{}".format(side)].shift(1),
            "HIT",
            "-",
        )

    tick_df["hit"] = np.where(
        (tick_df["pos_bound_hit"] == "HIT") | (tick_df["neg_bound_hit"] == "HIT"),
        tick_df["cusum_coord"],
        "no_hit",
    )

    hits = pd.DataFrame([x for x in tick_df["hit"] if x != "no_hit"])

    if len(hits) > 0:
        hits["abs_perc_change"] = abs((hits["price"] / hits["price"].shift(1) - 1) * 100)
        hits.dropna(inplace=True)
        return hits
    else:
        return pd.DataFrame(columns=["time", "price"])


@logger.catch
def main():

    btc_watch = Watcher("SOL/USDT", cperc=1, wait_sec=5)
    btc_watch.run()


if __name__ == "__main__":
    main()
