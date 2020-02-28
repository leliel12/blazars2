import os
import pickle
import sys
import pathlib
import argparse

import pandas as pd

import numpy as np

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

BIN_PATH = PATH / "bin"

COLUMNS_NO_FEATURES = ['id', 'tile', 'cnt', 'ra_k', 'dec_k']

COLUMNS_TO_REMOVE = [
    'scls_h', 'scls_j', 'scls_k', 'vs_type', 'vs_catalog',
    "AndersonDarling", "AmplitudeJ", "AmplitudeH", "AmplitudeJH", "AmplitudeJK",
    'Freq1_harmonics_rel_phase_0', 'Freq2_harmonics_rel_phase_0', 'Freq3_harmonics_rel_phase_0',
    "CAR_mean", "CAR_tau", "CAR_sigma", "StetsonK", "Meanvariance"]


def clean(df):
    df = df[[c for c in df.columns if c not in COLUMNS_TO_REMOVE]].copy()
    df["tile"] = df.id.apply(lambda i: "b" + str(i)[1:4])

    # clean
    df = df.dropna()

    df = df[
        (df.cnt >= 30) &
        df.c89_hk_color.between(-100, 100) &
        df.c89_jh_color.between(-100, 100) &
        df.c89_jk_color.between(-100, 100) &
        df.n09_hk_color.between(-100, 100) &
        df.n09_jh_color.between(-100, 100) &
        df.n09_jk_color.between(-100, 100)]

    # features columns
    features = [c for c in df.columns.values if c not in COLUMNS_NO_FEATURES]
    features.sort()

    # to float32
    df[features] = df[features].astype(np.float32)

    # reorder
    order = COLUMNS_NO_FEATURES + features
    df = df[order]

    df = df[~np.isinf(df.Period_fit.values)]
    df = df[~df.Gskew.isnull()]

    return df


def split(df, size):
    nums = int(len(df) / size)
    ids_split = np.array_split(df.id.values, nums)
    for idx, part in enumerate(ids_split):
        yield idx, df[df.id.isin(part)].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", dest="path", type=pathlib.Path)

    ns = parser.parse_args(sys.argv[1:])

    df = pd.read_pickle(ns.path)
    df = clean(df)

    name = ns.path.name.split(".", 1)[0]
    for idx, part in split(df, size=20_000):
        filename = BIN_PATH / f"{name}_-{idx}.csv.bz2"
        print(f"Writing {filename}")
        part.to_csv(filename, compression="bz2", index=False)



if __name__ == "__main__":
    main()
