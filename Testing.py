import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import KFold
from SepsisClassifier import *
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score
from lightgbm import plot_importance, plot_metric
import pickle as pk
import seaborn as sns

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV

warnings.filterwarnings(action='ignore', message='Mean of empty slice')


def plot_(X, y, ns_est: iter, iterations: iter, leaves: iter, boosts: iter):

    results = {"boost": [], "leaves": [], "n_est": [], "iterations": [], "f1": [], "fold": []}

    counter = -1
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        counter += 1
        print(f"Fold: {counter}")
        for n_est in ns_est:
            for leaf in leaves:
                for boost in boosts:
                    gb = LGBM(n_est=n_est, boost=boost, num_leaves=leaf)
                    fold_X_train = X[train]
                    fold_y_train = y.iloc[train]
                    fold_X_test = X[test]
                    fold_y_test = y.iloc[test]

                    gb.fit(fold_X_train, fold_y_train)
                    for iteration in iterations:

                        pred = gb(fold_X_test, iteration)

                        fold_f1 = f1_score(fold_y_test, pred)
                        results["boost"].append(boost)
                        results["leaves"].append(leaf)
                        results["n_est"].append(n_est)
                        results["iterations"].append(iteration)
                        results["f1"].append(fold_f1)
                        results["fold"].append(counter)

                # print(results)
    return pd.DataFrame.from_dict(results, "columns")


if __name__ == "__main__":

    train_path = "C:/Users/orper/PycharmProjects/Lab2HW1/train/"
    test_path = "C:/Users/orper/PycharmProjects/Lab2HW1/test/"

    # time0 = time.time()
    # train_dl = DataLoader(train_path, size=19999)
    # train_dl.X = train_dl.X.dropna(axis=1, how="all")
    # print(f"Loading Time: {time.time() - time0}")
    # print(len(train_dl.X.columns))
    # # #
    # time0 = time.time()
    # X = train_dl.impute_data(KNNImputer(n_neighbors=10, weights="uniform"), fit=True)
    # y = train_dl.y
    # print(f"Impute Time: {time.time() - time0}")
    # #
    # data = (train_dl, X)
    # #
    # with open('Titparea.pickle', 'wb') as f:
    #     pk.dump(data, f)
    #
    with open('Titparea.pickle', 'rb') as f:
        data = pk.load(f)

    # with open('data2.pickle', 'rb') as f:
    #     data = pk.load(f)
    train_dl, X = data

    # X = None
    #
    # X, y = data
    y = train_dl.y

    # baseline = BinaryClassifier(in_dims=len(train_dl.X.columns), hidden_dims=[100, 50, 20])

    # baseline = BinaryClassifier(in_dims=len(X.shape[1]), hidden_dims=[100, 50, 20])
    lgbm = LGBM(n_estimators=200, boosting_type="gbdt")

    print("Fitting BaseLine")
    # losses = baseline.fit(X, y, epochs=300, lr=0.001)
    print("Fitting LGBM")
    # print(X.shape)
    lgbm.fit(X, y, list(train_dl.X.columns))

    with open("model.pickle", "wb") as f:
        data = (lgbm, train_dl)
        pk.dump(data, f)
    # plot_metric(booster=lgbm.lgbm, metric="f1")
    # ax = plot_importance(lgbm.lgbm)
    # plt.show()

    model = SelectFromModel(lgbm, prefit=True)
    X_new = model.transform(X)
    print(model.get_feature_names_out(train_dl.X.columns))
    print(X_new.shape)

    lgbm2 = LGBM(n_estimators=200, boosting_type="gbdt")
    lgbm2.fit(X_new, y)

    model3 = RFECV(lgbm, scoring='f1', verbose=1, step=10)
    time0 = time.time()
    X_new = model3.fit_transform(X, y)
    print(X_new.shape)
    print(f"took {time.time() - time0}")

    lgbm3 = LGBM(n_estimators=200, boosting_type="gbdt")
    lgbm3.fit(X_new, y)


# X_new = SelectKBest(chi2, k=100).fit_transform(X, y)
    # print(X_new)

    test_dl = DataLoader(test_path, size=9999)



    #
    X = test_dl.impute_data(train_dl.impute)
    y = test_dl.y

    data = (test_dl, X)
    with open('Test.pickle', 'wb') as f:
        pk.dump(data, f)
    #
    # y_hat_baseline = baseline(X)
    # y_hat_baseline = torch.argmax(y_hat_baseline, dim=1)

    # y_hat_lgbm = lgbm2(model.transform(X))
    #
    # f1_baseline = f1_score(y, y_hat_baseline.detach().numpy())
    # f1_lgbm = f1_score(y, y_hat_lgbm)
    #
    # print(f"F1 BaseLine: {f1_baseline} | F1 LGBM: {f1_lgbm}")
    # print(f"F1 BaseLine: {f1_baseline}")
    # print(f"F1 LGBM FS: {f1_lgbm}")

    # y_hat_lgbm = lgbm(X)
    # f1_lgbm = f1_score(y, y_hat_lgbm)
    # print(f"F1 LGBM: {f1_lgbm}")
    #
    # y_hat_lgbm = lgbm3(model3.transform(X))
    # f1_lgbm = f1_score(y, y_hat_lgbm)
    # print(f"F1 LGBM FS CV: {f1_lgbm}")

    # print("Saving Results")
    # test_data = (test_dl, tX)
    # with open('test.pickle', 'wb') as f:
    #     pk.dump(test_data, f)

    # n_ests = [200]
    # iters = [10, 50, 100, 200, 500, 1000]
    # leaves = [31]
    # boost = ["dart", "gbdt"]
    #
    # results_df = plot_(X, y, n_ests, iters, leaves, boost)
    # print(results_df)
    #
    # sns.lineplot(data=results_df, x="iterations", y="f1", hue="boost")
    # plt.show()
