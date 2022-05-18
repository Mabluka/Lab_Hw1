from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from features import *
from lightgbm import LGBMClassifier
from sklearn.impute import KNNImputer
from SepsisClassifier import DataLoader
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.filterwarnings(action='ignore', message='Mean of empty slice')



if __name__ == "__main__":
    train_path = "C:/Users/orper/PycharmProjects/Lab2HW1/train/"
    n = 1000
    folds = 5
    to_shuffle = True

    neighbors = [1, 5, 10, 50]
    estimators = [100, 200, 300, 500]

    dl = DataLoader(train_path, n, load_file=None, save_file=None)
    dl.X = dl.X.dropna(axis=1, how="all")

    for nn in neighbors:
        X = dl.impute_data(KNNImputer(n_neighbors=nn, weights="uniform"), fit=True)
        for estimator in estimators:

            gb = LGBMClassifier(n_estimators=estimator, random_state=42, is_unbalance=True, boosting_type="dart")
            rf_mean = []

            kf = KFold(n_splits=folds, shuffle=to_shuffle)
            for train, test in kf.split(X):
                fold_X_train = X[train]
                fold_y_train = dl.y.iloc[train]
                fold_X_test = X[test]
                fold_y_test = dl.y.iloc[test]

                gb.fit(fold_X_train, fold_y_train)
                pred = gb.predict(fold_X_test)

                fold_f1_rf = f1_score(fold_y_test, pred)
                rf_mean.append(fold_f1_rf)

            print(f"K-neighbors = {nn}, N-estimators = {estimator} -> "
                  f"Mean = {np.mean(rf_mean)}, Var = {np.var(rf_mean)}")





# importences_df = pd.DataFrame.from_dict(importences, orient="index")
# importences_df["mean"] = importences_df.mean(axis=1)
# importences_df["sd"] = importences_df.std(axis=1)
# # plt.show()
#
# indices = np.argsort(importences_df["mean"].to_numpy())
# indices = indices[-20:]
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.barh(range(len(indices)), importences_df["mean"].to_numpy()[indices],
#          xerr=importences_df["sd"].to_numpy()[indices], align="center")
#
# plt.yticks(range(len(indices)), importences_df.index.to_numpy()[indices])
# plt.ylim([-1, len(indices)])
# plt.title(f"K-n:{nn},N-est:{estimator},B:{balanced}: "
#           f"M:{np.mean(rf_mean)}, V:{np.var(rf_mean)}")
# plt.tight_layout()
# plt.show()
# print(f"K-neighbors = {nn}, N-estimators = {estimator}, Balanced = {balanced} -> "
#       f"Mean = {np.mean(rf_mean)}, Var = {np.var(rf_mean)}")