from SepsisClassifier import *
import sys
import pickle as pk
import warnings
import zipfile

warnings.filterwarnings(action='ignore', message='Mean of empty slice')


if __name__ == "__main__":
    predict_path = sys.argv[1]
    file_name = "dl.zip"

    archive = zipfile.ZipFile(file_name, 'r')
    dl = archive.extract('dl.pickle')

    with open("model.pickle", "rb") as f:
        model = pk.load(f)

    with open("dl.pickle", "rb") as f:
        dl = pk.load(f)

    test_dl = DataLoader(predict_path)
    X = test_dl.impute_data(dl)

    y_hat = model(X)
    test_dl.X["SepsisLabel"] = y_hat
    test_dl.X["ID"] = test_dl.ids

    predict_df = test_dl.X[["ID", "SepsisLabel"]]

    predict_df.to_csv("prediction.csv")