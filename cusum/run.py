from datetime import datetime, timedelta
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.ftx_rest import ftx as FTX


def detect_hits(df, cusum_thresh):
    points = df.copy()

    points["cusum_coord"] = points[["time", "price"]].to_dict("records")

    points["diff"] = (points["price"] / points["price"].shift(1) - 1.0) * 100
    points.dropna(inplace=True)

    points["diff_pos"] = np.where(points["diff"] > 0.0, points["diff"], 0.0)
    points["diff_neg"] = np.where(points["diff"] < 0.0, points["diff"] * -1, 0.0)

    for side in ["pos", "neg"]:
        points["cusum_{}".format(side)] = points["diff_{}".format(side)].cumsum()
        points["cusum_{}".format(side)] = points["cusum_{}".format(side)] % cusum_thresh
        points["{}_bound_hit".format(side)] = np.where(
            points["cusum_{}".format(side)] < points["cusum_{}".format(side)].shift(1), "HIT", "-"
        )

    points["hit"] = np.where(
        (points["pos_bound_hit"] == "HIT") | (points["neg_bound_hit"] == "HIT"),
        points["cusum_coord"],
        "no_hit",
    )

    hits = pd.DataFrame([x for x in points["hit"] if x != "no_hit"])
    hits["abs_perc_change"] = abs((hits["price"] / hits["price"].shift(1) - 1) * 100)
    hits.dropna(inplace=True)

    return hits


def main():

    ftx = FTX()

    now = int(datetime.now().timestamp() * 1000)
    then = now - 1000 * 60 * 60 * 4

    points = pd.DataFrame([x["info"] for x in ftx.fetch_trades("BTC/USDT", since=then, until=now)])
    points["price"] = points["price"].astype(float)

    hit_times = detect_hits(points, cusum_thresh=0.75)
    print(hit_times["abs_perc_change"].describe())
    exit()

    fig1 = go.Scatter(x=points["time"], y=points["price"])
    fig2 = go.Scatter(x=hit_times["time"], y=hit_times["price"], mode="markers")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(fig1)
    fig.add_trace(fig2, secondary_y=True)

    fig.show()


if __name__ == "__main__":
    main()
