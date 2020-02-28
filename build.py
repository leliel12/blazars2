import os
import pickle
import pathlib

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import joblib


PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

BIN_PATH = PATH / "bin"

DATA_PATH = PATH / "_data"

NO_FEATURES = ['id', 'tile', 'cnt', 'ra_k', 'dec_k']


def create_dir(path):
    path = pathlib.Path(path)
    if path.is_dir():
        raise IOError("Please remove the directory {}".format(path))
    os.makedirs(str(path))


def scale(df):
    df = df.copy()
    features = [c for c in df.columns.values if c not in NO_FEATURES]

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features].values)

    return scaler, df


def build():
    create_dir(DATA_PATH)

    by_tile = {}
    tile_outs, scaled_outs, scalers_outs = {}, {}, {}
    for path in BIN_PATH.glob("*.csv.bz2"):
        tilename = path.name.split("_", 2)[1]

        if tilename not in by_tile:
            by_tile[tilename] = []
            tile_outs[tilename] = path.name.split("_-", 1)[0] + ".pkl.bz2"
            scaled_outs[tilename] = (
                path.name.split("_-", 1)[0] + "_scaled.pkl.bz2")
            scalers_outs[tilename] = (
                "scaler_" + path.name.split("_-", 1)[0] + ".pkl.bz2")
        by_tile[tilename].append(path)

    merged = {}
    for tile, parts in by_tile.items():
        print(f">>> TILE {tile}")
        tile_outpath = DATA_PATH / tile_outs[tile]
        scaled_outpath = DATA_PATH / scaled_outs[tile]
        scaler_outpath = DATA_PATH / scalers_outs[tile]

        merged = None
        for p in parts:
            print(f"   !!! Reading {p}")
            df = pd.read_csv(p)
            if merged is None:
                merged = df
            else:
                merged = pd.concat([merged, df], ignore_index=True)

        print(f"   !!! Scaling")
        scaler, scaled = scale(merged)

        print(f"   <<< Writing {tile_outpath}")
        joblib.dump(merged, tile_outpath, compress=3)

        print(f"   <<< Writing {scaled_outpath}")
        joblib.dump(scaled, scaled_outpath, compress=3)

        print(f"   <<< Writing {scaler_outpath}")
        joblib.dump(scaler, scaler_outpath, compress=3)

        print(" --- --- --- ")


if __name__ == "__main__":
    build()
