import numpy as np
import pandas as pd
from SepsisClassifier import *
import seaborn as sns
import pickle as pk
from matplotlib import pyplot as plt
import tabulate

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV



def histogram_plot(df, col, hue=None, ax=None, label=False, **kwargs):
    if ax is None:
        return sns.histplot(df, x=col, hue=hue, label=label, **kwargs)
    else:
        return sns.histplot(ax=ax, data=df, x=col, hue=hue, label=label, **kwargs)


def kde_plot(df, col, hue=None, ax=None, label=False, **kwargs):
    if ax is None:
        return sns.kdeplot(data=df, x=col, hue=hue, label=label, **kwargs)
    else:
        return sns.kdeplot(ax=ax, data=df, x=col, hue=hue, label=label, **kwargs)


def before_after_imputation(dl, X, col, split_by_label=False, log_scale=False):
    df_after = pd.DataFrame(X)
    df_after.columns = dl.X.columns

    df_before = dl.X.copy()
    df_before[col + "_after_imputation"] = df_after[col]
    df_before["Diag"] = dl.y

    hue = "Diag" if split_by_label else None
    # fig, axes = plt.subplots()
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1_ax = histogram_plot(df_before, col=col, hue=hue, ax=ax1, label=True, log_scale=log_scale)
    props = dict(boxstyle='round', facecolor='snow', alpha=0.7, ec="gainsboro")
    # place a text box in upper left in axes coords
    ax1.text(0.05, 0.95, "Before Imputation", transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    ax2 = fig.add_subplot(2, 2, 3)
    ax2_ax = histogram_plot(df_before, col=col + "_after_imputation", hue=hue, ax=ax2, label=True, log_scale=log_scale)
    props = dict(boxstyle='round', facecolor='snow', alpha=0.7, ec="gainsboro")
    # place a text box in upper left in axes coords
    ax2.text(0.05, 0.95, "After Imputation", transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    ax2.set_xlabel(col)


    ax3 = fig.add_subplot(1, 2, 2)
    ax3_ax = kde_plot(df_before, col=col, ax=ax3, label="Before Imputation", color="skyblue", log_scale=log_scale)
    ax3_ax2 = kde_plot(df_before, col=col + "_after_imputation", ax=ax3, label="After Imputation", log_scale=log_scale)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    ax3.legend()

    fig.suptitle(col + ": Histogram and KDE Before and After Imputation", fontsize=20, fontweight="normal",
                 fontname="STIXGeneral")
    plt.tight_layout()
    plt.savefig("Histogram" + col + ".png")
    plt.show()


def plot_feature_importance(ax, model_name, importance, names, k=None, feature_selection=None):
    if k is None:
        k = len(names)

    importance_df = {}
    for f_name, f_importance in zip(names, importance):
        importance_df[f_name] = f_importance

    importance_df = pd.DataFrame.from_dict(importance_df, "index")
    importance_df.columns = ["value"]

    indices = np.argsort(importance_df["value"].to_numpy())
    indices = indices[-k:]

    # Plot the feature Importance of the model

    ax.barh(range(len(indices)), importance_df["value"].to_numpy()[indices], align="center")

    for i in ax.patches:
        ax.text(i.get_width()+0.2, i.get_y()+0.2,
                 str(round((i.get_width()), 2)),
                 fontsize = 8, fontweight ='normal',
                 color ='black', alpha=0.7)

    ax.grid(b=True, color='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha=0.2)

    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)

    # ax.yticks(range(len(indices)), importance_df.index.to_numpy()[indices])  # need to set on ax1
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(importance_df.index.to_numpy()[indices], fontsize=8)
    ax.set_ylim([-1, len(indices)])  # need to set on ax1
    ax.tick_params(labelsize=8)

    if feature_selection is not None:
        ax.text(0.5, 0.2, model_name, transform=ax.transAxes, fontsize=20, alpha=1)
        ax.text(0.52, 0.15, f"{feature_selection} Feature Selection", transform=ax.transAxes, fontsize=12, alpha=1)
        ax.text(0.53, 0.12, f"(Best {k} features)", transform=ax.transAxes, fontsize=8, alpha=1)
    else:
        ax.text(0.5, 0.2, model_name, transform=ax.transAxes, fontsize=20, alpha=1)
        ax.text(0.52, 0.15, f"(Best {k} features)", transform=ax.transAxes, fontsize=12, alpha=1)

    return ax


def feature_importances(dart_features, dart_values, gbdt_features, gbdt_values, k, selection_method=None):
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1_ax = plot_feature_importance(ax1, "Dart", dart_values, dart_features, k, feature_selection="No")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2_ax = plot_feature_importance(ax2, "Gbdt", gbdt_values, gbdt_features, k, feature_selection="No")

    plt.tight_layout()
    plt.show()


def plot_importance(dart, gbdt, k, selection_method=None):
    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(3, 2, 1)
    importance_tuple = dart[0]
    ax1_ax = plot_feature_importance(ax1, "Dart", importance_tuple[1], importance_tuple[0], k, feature_selection="No")

    ax3 = fig.add_subplot(3, 2, 3)
    importance_tuple = dart[1]
    ax3_ax = plot_feature_importance(ax3, "Dart", importance_tuple[1], importance_tuple[0], k,
                                     feature_selection="RFECV")

    ax5 = fig.add_subplot(3, 2, 5)
    importance_tuple = dart[2]
    ax5_ax = plot_feature_importance(ax5, "Dart", importance_tuple[1], importance_tuple[0], k,
                                     feature_selection="Model Selection")


    ax2 = fig.add_subplot(3, 2, 2)
    importance_tuple = gbdt[0]
    ax2_ax = plot_feature_importance(ax2, "GBDT", importance_tuple[1], importance_tuple[0], k, feature_selection="No")

    ax4 = fig.add_subplot(3, 2, 4)
    importance_tuple = gbdt[1]
    ax4_ax = plot_feature_importance(ax4, "GBDT", importance_tuple[1], importance_tuple[0], k,
                                     feature_selection="RFECV")

    ax6 = fig.add_subplot(3, 2, 6)
    importance_tuple = gbdt[2]
    ax6_ax = plot_feature_importance(ax6, "GBDT", importance_tuple[1], importance_tuple[0], k,
                                      feature_selection="Model Selection")


    plt.tight_layout()
    plt.savefig("Feature_Importance.svg")
    plt.savefig("Feature_Importance.png")
    plt.show()







# Load The Processed Data
with open('Titparea.pickle', 'rb') as f:
    data = pk.load(f)
    dl, X = data
# dl.X["Diag"] = dl.y




tp = dl.X

# tp["HR_maxNf_I"] = X[:, 0]
# fig, axes = plt.subplots(2, 1, sharex="all", sharey="all", figsize=(10, 10))
# ax1 = histogram_plot(tp, col="HR_max24f", hue="Diag", label=True, ax=axes[0], kde=True, log_scale=True)
# ax2 = histogram_plot(tp, col="HR_max24f_I", hue="Diag", label=False, ax=axes[1], kde=True, log_scale=True)
# plt.tight_layout()
# plt.show()


# lgbm = LGBM(200)
# print(len(X))
# lgbm.fit(X, dl.y, names=list(dl.X.columns))
#
# model = SelectFromModel(lgbm, prefit=True)
# X_new = model.transform(X)
#
#
# lgbm1_1 = LGBM(n_estimators=200)
# lgbm1_1.fit(X_new, dl.y, names=list(model.get_feature_names_out(dl.X.columns)))
#
# model3 = RFECV(lgbm, scoring='f1', verbose=1, step=50)
#
# X_new = model3.fit_transform(X, dl.y)
#
# lgbm1_2 = LGBM(n_estimators=200)
# lgbm1_2.fit(X_new, dl.y, names=list(model3.get_feature_names_out(dl.X.columns)))
#
# lgbm2 = LGBM(200, boosting_type="gbdt")
# lgbm2.fit(X, dl.y, names=list(dl.X.columns))
#
# model = SelectFromModel(lgbm2, prefit=True)
# X_new = model.transform(X)
#
# lgbm2_1 = LGBM(n_estimators=200, boosting_type="gbdt")
# lgbm2_1.fit(X_new, dl.y, names=list(model.get_feature_names_out(dl.X.columns)))
#
# model3 = RFECV(lgbm2, scoring='f1', verbose=1, step=50)
#
# X_new = model3.fit_transform(X, dl.y)
#
# lgbm2_2 = LGBM(n_estimators=200, boosting_type="gbdt")
# lgbm2_2.fit(X_new, dl.y, names=list(model3.get_feature_names_out(dl.X.columns)))
#
#
# dart = [(lgbm.feature_names, lgbm.feature_importances_), (lgbm1_1.feature_names, lgbm1_1.feature_importances_),
#         (lgbm1_2.feature_names, lgbm1_2.feature_importances_)]
#
# gbdt = [(lgbm2.feature_names, lgbm2.feature_importances_), (lgbm2_1.feature_names, lgbm2_1.feature_importances_),
#         (lgbm2_2.feature_names, lgbm2_2.feature_importances_)]

#
# feature_importances(lgbm.feature_names, lgbm.feature_importances_,
#                     lgbm2.feature_names, lgbm2.feature_importances_, 20,
#                     "Test")

# plot_importance(dart, gbdt, 20)


before_after_imputation(dl, X, "PaCO2_maxNf", split_by_label=True)
#
before_after_imputation(dl, X, "Calcium_meanNf", split_by_label=True)



# log = False
# fill = True
# kde = False
#
# ax1 = histogram_plot(tp, col="HR_max24f", label="Before", log_scale=log, fill=fill, kde=kde)
# ax2 = histogram_plot(tp, col="HR_max24f_I", label="After", log_scale=log, color="skyblue", fill=fill, kde=kde)
# plt.tight_layout()
# plt.legend()
# plt.show()
#
#
# log = False
# fill = True
# ax1 = kde_plot(tp, col="HR_max24f", label="Before", log_scale=log)
# ax2 = kde_plot(tp, col="HR_max24f_I", label="After", log_scale=log, color="skyblue")
# plt.tight_layout()
# plt.legend()
# plt.show()




