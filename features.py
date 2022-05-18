import time

import numpy as np
from typing import Union, List, Callable, Optional, Tuple, Dict, Any


def window_features(df, cols, window: Optional[Union[List[int], int]] = None):


    def get_for_window(window_df, w):
        to_return = []
        maxf = np.nanmax(window_df, axis=0)
        minf = np.nanmin(window_df, axis=0)
        meanf = np.nanmean(window_df, axis=0)

        to_return += zip(map(lambda x: x + f"_max{w}f", cols), maxf)
        to_return += zip(map(lambda x: x + f"_min{w}f", cols), minf)
        to_return += zip(map(lambda x: x + f"_mean{w}f", cols), meanf)
        to_return += zip(map(lambda x: x + f"_maxmin{w}f", cols), maxf - minf)
        return to_return

    results = []
    fdf = df[cols].to_numpy()
    if window is None:
        return get_for_window(fdf, "N")
    for w in window:
        fdf = fdf[-w:]
        results += get_for_window(fdf, w)

    return results


def null_things_feature(df, cols):
    results = []
    # nans = df[cols].isnull().sum(axis=0).to_numpy()
    nans = np.isnan(df[cols].to_numpy()).sum(axis=0)

    results += zip(map(lambda x: x + "_cnf", cols), nans)
    results += zip(map(lambda x: x + "_fnf", cols), nans/len(df))
    return results


def last_values(df, cols):
    fdf = df[cols].to_numpy()[-1]
    return zip(map(lambda x: x + "_lvf", cols), fdf)




