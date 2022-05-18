import collections

import pandas as pd
import numpy as np
import seaborn as sn
import os
from mpl_toolkits.mplot3d import Axes3D

from typing import Union, List, Callable, Optional, Tuple, Dict

#  TODO: Do Data Analysis

# Need to extract as much data as possible together
from matplotlib import pyplot as plt

data_path = "C:/Users/orper/PycharmProjects/Lab2HW1/train/patient_"


def trim(patient_df):
    patient_diag = patient_df["SepsisLabel"].values
    diagnosed = 1 in patient_diag
    if diagnosed:
        index = patient_diag.argmax()
    else:
        index = len(patient_df)
    trimed_df = patient_df.iloc[:index + 1, :].drop(columns="SepsisLabel")
    return trimed_df, diagnosed



def acquire_data(path, n, columns):
    data = {}
    data2 = collections.defaultdict(list)
    for i in range(n):
        p_data = {}
        i_df, i_diag = trim(pd.read_csv(path + f"{i}.psv",sep='|'))
        # data[i, "mean"] = i_df.mean()
        for col in columns:
            p_data[col + "_d2"] = i_df[col].iloc[-1] - i_df[col].iloc[-2] if len(i_df[col]) >= 2 else np.nan
            p_data[col + "_m"] = i_df[col].mean()
            p_data[col + "_max"] = i_df[col].max()
            p_data[col + "_min"] = i_df[col].min()
            p_data[col + "_m10"] = i_df[col][-10:].mean()
            p_data[col + "_m12"] = i_df[col][-12:].mean()
            p_data[col + "_m24"] = i_df[col][-24:].mean()
            p_data[col + "_nan"] = np.count_nonzero(np.isnan(i_df[col].to_numpy()))
            p_data[col + "_cnt"] = len(i_df)
            data2[col] += list(i_df[col].to_numpy())
        p_data["Tn"] = i_df["ICULOS"].iloc[-1]
        p_data["Gender"] = "Male" if i_df["Gender"].iloc[-1] else "Female"
        p_data["Gender_Marker"] = 'o' if p_data["Gender"] else 's'
        p_data["Age"] = i_df["Age"].iloc[-1]
        p_data["Label"] = i_diag
        data[i] = p_data

    data = pd.DataFrame.from_dict(data, "index")
    data2 = pd.DataFrame.from_dict(data2)
    return data, data2


def plot_histograms(df, col, hue=None):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
    sn.histplot(ax=axes[0][0], data=df, x=col + "_min", kde=True, hue=hue, legend=False)

    sn.histplot(ax=axes[0][1], data=df, x=col + "_m", kde=True, hue=hue, legend=False)

    sn.histplot(ax=axes[0][2], data=df, x=col + "_max", kde=True, hue=hue, legend=True)

    sn.histplot(ax=axes[1][0], data=df, x=col + "_m6", kde=True, hue=hue, legend=False)

    sn.histplot(ax=axes[1][1], data=df, x=col + "_m12", kde=True, hue=hue, legend=False)

    sn.histplot(ax=axes[1][2], data=df, x=col + "_m24", kde=True, hue=hue, legend=False)

    fig.tight_layout()
    plt.show()


def plot_boxplots(df, y, x):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True, sharex=True)
    sn.boxplot(ax=axes[0][0], data=df, y=y + "_min", x=x)

    sn.boxplot(ax=axes[0][1], data=df, y=y + "_m", x=x)

    sn.boxplot(ax=axes[0][2], data=df, y=y + "_max", x=x)

    sn.boxplot(ax=axes[1][0], data=df, y=y + "_m6", x=x)

    sn.boxplot(ax=axes[1][1], data=df, y=y + "_m12", x=x)

    sn.boxplot(ax=axes[1][2], data=df, y=y + "_m24", x=x)

    fig.tight_layout()
    plt.show()


def plot_hist_box(df, y, x):
    fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharey="row")
    ax = sn.histplot(ax=axes[0][0], data=df, x=y + "_min", kde=True, hue=x, legend=False, stat="density")

    sn.histplot(ax=axes[0][1], data=df, x=y + "_m", kde=True, hue=x, legend=False, stat="density")

    sn.histplot(ax=axes[0][2], data=df, x=y + "_max", kde=True, hue=x, legend=True, stat="density")

    sn.histplot(ax=axes[1][0], data=df, x=y + "_m10", kde=True, hue=x, legend=False, stat="density")

    sn.histplot(ax=axes[1][1], data=df, x=y + "_m12", kde=True, hue=x, legend=False, stat="density")

    sn.histplot(ax=axes[1][2], data=df, x=y + "_m24", kde=True, hue=x, legend=False, stat="density")

    ax = sn.boxplot(ax=axes[2][0], data=df, y=y + "_min", x=x)
    ax.set(xlabel=None)
    ax.set(xticklabels=[])

    ax = sn.boxplot(ax=axes[2][1], data=df, y=y + "_m", x=x)
    ax.set(xlabel=None)
    ax.set(xticklabels=[])

    ax = sn.boxplot(ax=axes[2][2], data=df, y=y + "_max", x=x)
    ax.set(xlabel=None)
    ax.set(xticklabels=[])

    ax = sn.boxplot(ax=axes[3][0], data=df, y=y + "_m10", x=x)
    ax.set(xticklabels=[])

    ax = sn.boxplot(ax=axes[3][1], data=df, y=y + "_m12", x=x)
    ax.set(xticklabels=[])

    ax = sn.boxplot(ax=axes[3][2], data=df, y=y + "_m24", x=x)
    ax.set(xticklabels=[])

    fig.tight_layout()
    plt.show()


def scatter3dheat(df, x, y, z, label=None):
    norm = plt.Normalize(df[z].min(), df[z].max())
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])
    ax = sn.scatterplot(data=df, x=x, y=y, hue=z, style=label, palette="flare")
    ax.get_legend().remove()
    ax.figure.colorbar(sm, label=z)
    plt.show()


def scatter3d_time(df, x, y, zs, label=None):
    fig, axes = plt.subplots(len(zs) // 3 + 1 * (len(zs) % 3 > 0), 3, figsize=(15, 10), sharex=True, sharey=True)
    for index_, z in enumerate(zs):
        z = z + "_m"
        norm = plt.Normalize(df[z].min(), df[z].max())
        sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
        sm.set_array([])
        ax = sn.scatterplot(ax=axes[index_ // 3][index_ % 3], data=df, x=x, y=y, hue=z, style=label, palette="flare")
        ax.get_legend().remove()
        ax.figure.colorbar(sm, label=z, ax=axes[index_ // 3][index_ % 3])
    fig.tight_layout()
    plt.show()

def scatter2d_time(df, x, ys, label=None):
    x = x + "_m"
    fig, axes = plt.subplots(len(ys) // 3 + 1 * (len(ys) % 3 > 0), 3, figsize=(15, 10), sharex=True)
    for index_, y in enumerate(ys):
        y = y + "_m"
        ax = sn.scatterplot(ax=axes[index_ // 3][index_ % 3], data=df, x=x, y=y, hue=label)
    fig.tight_layout()
    plt.show()

def scatter_time_same(df, x, y, vars: list = None, hue=None):
    if vars is None:
        vars = ["_m", "_m12", "_d2"]
    g = sn.PairGrid(data=df, x_vars=[x + var for var in vars], y_vars=[y + var for var in vars], hue=hue)
    g.map(sn.scatterplot)
    plt.show()



def scatter_time_different(df, x, y, hue=None):
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True, sharex=True)
    sn.scatterplot(ax=axes[0][0], data=df, x=x + "_min", y=y + "_min", hue=hue, legend=False)

    sn.histplot(ax=axes[0][1], data=df, x=x + "_m", y=y + "_m", hue=hue, legend=False)

    sn.histplot(ax=axes[0][2], data=df, x=x + "_max", y=y + "_max", hue=hue, legend=True)

    sn.histplot(ax=axes[1][0], data=df, x=x + "_m6", y=y + "_m6", hue=hue, legend=False)

    sn.histplot(ax=axes[1][1], data=df, x=x + "_m12", y=y + "_m12", hue=hue, legend=False)

    sn.histplot(ax=axes[1][2], data=df, x=x + "_m24", y=y + "_m24", hue=hue, legend=False)

    fig.tight_layout()
    plt.show()


columns = ["HR", "SBP", "DBP", "Temp", "O2Sat", "Resp", "MAP", "WBC", "FiO2", "pH", "AST", "PaCO2", "PTT",
           "Bilirubin_total"]
data_dict, data_dict2 = acquire_data(data_path, 19999, columns)

for col in columns:
    # plot_histograms(data_dict, col, "Label")
    # plot_boxplots(data_dict, col, "Label")
    plot_hist_box(data_dict, col, "Label")


true_age = data_dict[data_dict["Label"] == 1]["Age"].mean()
print(true_age)
false_age = data_dict[data_dict["Label"] == 0]["Age"].mean()
print(false_age)

# sn.catplot(data=data_dict, x="Label", y="Age")
# plt.show()

# seen_ = set()
# for col1 in columns:
#     seen_.add(col1)
#     scatter2d_time(data_dict, col1, set(columns) - {col1}, "Label")


# seen_ = []
# for col1 in columns:
#     for col2 in columns:
#         if col1 != col2 and col2 not in seen_:
#             scatter_time_same(data_dict, col1, col2, hue="Label", vars=["_m"])
#     seen_.append(col1)


# seen_ = set()
# for col1 in columns:
#     seen_.add(col1)
#     for col2 in columns:
#         if col1 != col2 and col2 not in seen_:
#             scatter3d_time(data_dict, col1 + "_m", col2 + "_m", set(columns) - {col1, col2}, "Label")



# scatter3dheat(data_dict, "HR_m", "SBP_m", "DBP_m")
# scatter3dheat(data_dict, "HR_m", "SBP_m", "DBP_m", "Label")
# plot_boxplots(data_dict, "HR", "Label")
#
# fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey=True)
# sn.histplot(ax=axes[0][0], data=data_dict, x="HR_min", kde=True, hue="Label", legend=False)
#
# sn.histplot(ax=axes[0][1], data=data_dict, x="HR_m", kde=True, hue="Label", legend=False)
#
# sn.histplot(ax=axes[0][2], data=data_dict, x="HR_max", kde=True, hue="Label", legend=True)
#
# sn.histplot(ax=axes[1][0], data=data_dict, x="HR_m6", kde=True, hue="Label", legend=False)
#
# sn.histplot(ax=axes[1][1], data=data_dict, x="HR_m12", kde=True, hue="Label", legend=False)
#
# sn.histplot(ax=axes[1][2], data=data_dict, x="HR_m24", kde=True, hue="Label", legend=False)
# # fig.legend(axes[0].get_legend().legendHandles, ["False", "True"], title="Has Sepsis")
# # axes[0].get_legend().remove()
# fig.tight_layout()
# plt.show()
#
# sn.scatterplot(data=data_dict, x="Resp_d2", y="HR_d2", hue="Label", style="Gender", s=10)
# plt.show()

# data_dict = data_dict.drop(columns=["Gender_Marker"])
# corr = data_dict.corr()
# sn.heatmap(corr)
# plt.show()

# sn.lineplot(data=data_dict, x="Last_ICULOS", y="Temp_mean", hue="Label")
# plt.show()
# print(data_dict)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# x = data_dict["HR_d2"]
# y = data_dict["Tn"]
# z = data_dict["SBP_m12"]
#
# ax.scatter(x, y, z)
# plt.show()
#
# sn.scatterplot(data=data_dict, x="HR_d2", y="SBP_d2", hue="Label")
# plt.show()

#
# sn.boxplot(data=data_dict, x="Label", y="HR_d2")
# plt.show()
#
# sn.boxplot(data=data_dict, x="Label", y="HR_m12")
# plt.show()
#
# sn.boxplot(data=data_dict, x="Label", y="Temp_m12")
# plt.show()
#
# sn.boxplot(data=data_dict, x="Label", y="Tn")
# plt.show()
#
# sn.boxplot(data=data_dict, x="Label", y="SBP_m12")
# plt.show()
#
# sn.boxplot(data=data_dict, x="Label", y="DBP_m12")
# plt.show()
#
# sn.boxplot(data=data_dict, x="Label", y="Resp_m12")
# plt.show()
#
# #
# sn.scatterplot(data=data_dict2, x="HR", y="Temp")
# plt.show()
#
# sn.scatterplot(data=data_dict2, x="HR", y="SBP")
# plt.show()
#
# sn.scatterplot(data=data_dict2, x="HR", y="MAP")
# plt.show()
#
# sn.scatterplot(data=data_dict2, x="SBP", y="MAP")
# plt.show()
#
# sn.scatterplot(data=data_dict2, x="SBP", y="O2Sat")
# plt.show()
#
#
# #
# sn.displot(data=data_dict2, x="HR", kde=True)
# plt.show()
#
# sn.displot(data=data_dict, x="HR_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="HR_min", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="HR_m", y="SBP_m", kind="kde", hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="HR_max", y="Temp_min", kind="kde", hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="SBP_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="Temp_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="Resp_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="pH_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="MAP_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="O2Sat_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="WBC_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="FiO2_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="AST_m", kde=True, hue="Label")
# plt.show()
#
# sn.displot(data=data_dict, x="Age", kde=True, hue="Label")
# plt.show()




