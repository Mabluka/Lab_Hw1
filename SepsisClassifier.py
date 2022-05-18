import warnings
import os
import time

import numpy as np
import pandas as pd
from features import *

from lightgbm import LGBMClassifier
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.feature_selection import RFE

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')


class DataLoader:
    """
    Loader class to load, process and handle the psv files

    Attributes
    __________
    **X** : `pd.DataFrame`
        Patients features df
    **y** : `pd.Series`
        Patients Labels
    **ids** : `pd.Series`
        Patients ID's

    Methods
    _______
    impute_data(impute, fit = false)
        Impute variable ``X`` to fill missing values.
    """

    def __init__(self, path: str, size: Optional[int] = None,
                 load_file: Optional[str] = None, save_file: Optional[str] = None
                 ):

        # cols = ["HR", "O2Sat", "Temp", "Resp", "MAP", "SBP", "WBC", "Creatinine", "Platelets", "FiO2", "SaO2", "PTT", "BUN", "Calcium",
        #         "Phosphate", "Hct", "Lactate", "Alkalinephos", "Glucose", "Hgb", "Age", "Gender", "HospAdmTime", "ICULOS"]

        cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
                "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
                "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
                "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
                "Age", "Gender", "HospAdmTime", "ICULOS", "Unit1", "Unit2", "Age", "Gender", "HospAdmTime", "ICULOS"]
        if load_file is None:
            _data = self.__load_psv_multiple(path, size, cols)
        else:
            _data = pd.read_csv(load_file, nrows=size, usecols=cols)

        if save_file is not None:
            _data.to_csv(save_file)
        self.impute = None
        self.X, self.y, self.ids = self.__split_df(_data)


    @staticmethod
    def __load_psv_single(path: str, cols: Optional[Union[List[str], str]]) -> Tuple[pd.DataFrame, int]:
        """
        Load a single psv file from start to the first 1 in column 'SepsisLabel'

        Parameters
        __________
        **path**: `str`
            Path to the psv file

        **cols**: `list` [`str`] or `str`, optional
            Columns to load from the psv

        Returns
        _______
        **trimmed_df**: `pd.DataFrame`
            A trimmed pandas dataframe containing the rows until first '1' in column 'SepsisLabel'
        **label**: `int`
            Label of patient; 1 if there exists 1 in column 'SepsisLabel' else 0
        """
        if cols is not None:
            cols += ['SepsisLabel']

        patient_df = pd.read_csv(path, sep='|', usecols=cols)
        patient_sepsis = patient_df["SepsisLabel"].values

        label = 1 in patient_sepsis
        if label:
            index = patient_sepsis.argmax()  # Find first index of 1 in patient_sepsis
        else:
            index = len(patient_df)

        # Trim df to row index and remove SepsisLabel Column
        trimmed_df = patient_df.iloc[:index + 1, :].drop(columns="SepsisLabel")
        return trimmed_df, label

    @staticmethod
    def __extract_features(df: pd.DataFrame) -> \
            Dict[str, float]:
        """
        Extract features from the patient dataframe

        Parameters
        __________
        **df** : `pd.DataFrame`
            The patient dataframe
        **features** : `list` [callable] , optional
            Feature functions to apply takes 1 argument:

            - ``df``: dataframe to operate on (`pd.DataFrame`)

        Returns
        _______
        **features_dict** : `dict` [`str`, `float`]
            Dict containing the new features and their values for the patient
        """

        features_dict = {}

        # window_feature_cols = ["HR", "O2Sat", "Temp", "Resp", "MAP", "SBP", "WBC", "Creatinine", "Platelets", "FiO2", "SaO2", "PTT", "BUN", "Calcium",
        #                        "Phosphate", "Hct", "Lactate", "Alkalinephos", "Glucose", "Hgb"]

        window_feature_cols = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
                               "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine",
                               "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
                               "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
        null_things_cols = window_feature_cols
        last_values_cols = ["Age", "Gender", "HospAdmTime", "ICULOS"]

        # features_dict.update(window_features(df, cols=window_feature_cols, window=[24, 12]))
        features_dict.update(window_features(df, cols=window_feature_cols))

        features_dict.update(null_things_feature(df, null_things_cols))

        features_dict.update(last_values(df, last_values_cols))

        return features_dict

    def __load_psv_multiple(self, dir_path: str, size: Optional[int] = None,
                            cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load a multiple amount of psv files from a specific directory and extract the features to create a new dataframe

        Parameters
        __________
        **dir_path** : `str`
            The path to the directory of the psv files
        **size** : `int`, optional
            The amount of psv files to load
        **features** : `list` [callable], optional
            Features functions to apply
        **cols** : `list` [`str`], optional
            Columns to load from the psv files

        Returns
        _______
        **_data** : `pd.DataFrame`
            The new dataframe of all the features extracted from the psv files

        """
        pd_dict = {}
        counter = 0
        for filename in os.listdir(dir_path):
            if size is not None and counter > size:
                break
            f = os.path.join(dir_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                trimmed_df, label = self.__load_psv_single(f, cols)
                features_dict = self.__extract_features(trimmed_df)

                features_dict["ID"] = filename
                features_dict["Diag"] = int(label)

                pd_dict[counter] = features_dict
                counter += 1
        _data = pd.DataFrame.from_dict(pd_dict, "index")
        return _data

    @staticmethod
    def __split_df(df: pd.DataFrame):
        """
        Split the dataframe to X, y and ids

        Parameters
        __________
        **df** : `pd.DataFrame`
            The dataframe to split

        Returns
        _______
        **X** : `pd.DataFrame`
            The features dataframe
        **y** : `pd.Series`
            The label column of df
        **ids** : `pd.Series`
            The ID column of df
        """
        X = df.iloc[:, :-2]
        y = df.iloc[:, -1]
        ids = df.iloc[:, -2]
        return X, y, ids

    def impute_data(self, impute, fit: bool = False) -> pd.DataFrame:
        """
        Impute missing data in features

        Parameters
        __________
        **impute** : `KNNImputer`
            KNN imputation
        **fit** : `bool`
            If shall fit or only transform

        Returns
        _______
        **x** : `pd.DataFrame`
            The X dataframe after imputation
        """
        x = self.X
        if fit:
            x = impute.fit_transform(x)
            self.impute = impute
        else:
            x = impute.transform(x)

        return x


class LGBM:
    def __init__(self, n_estimators: int = 200, boosting_type: str = "dart", num_leaves: int = 32,
                 learning_rate: float = 0.1, is_unbalance: bool = True, **kwargs):
        self.lgbm = LGBMClassifier(n_estimators=n_estimators, is_unbalance=is_unbalance,
                                   boosting_type=boosting_type,
                                   num_leaves=num_leaves, learning_rate=learning_rate, **kwargs)

    def fit(self, X, y, names="auto"):
        self.lgbm.fit(X, y, feature_name=names)

    def predict(self, X, num_iter=None):
        return self.lgbm.predict(X, num_iteration=num_iter)

    def __call__(self, X, num_iter=None):
        return self.predict(X, num_iter)

    def get_params(self, deep):
        return self.lgbm.get_params(deep)

    @property
    def feature_importances_(self):
        return self.lgbm.feature_importances_

    @property
    def feature_names(self):
        return self.lgbm.feature_name_



class BinaryClassifier(nn.Module):
    def __init__(self, in_dims:int,  hidden_dims: list):
        super(BinaryClassifier, self).__init__()

        layers = []

        in_ = in_dims
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_, h_dim))
            layers.append(nn.ReLU())
            in_ = h_dim
        layers.append(nn.Linear(in_, 2))
        layers.append(nn.Sigmoid())
        # layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        X = torch.tensor(X)
        X = X.float()
        # X = X[None, :]
        return self.layers(X)

    def fit(self, X, y, epochs=100, lr=0.01):
        self.float()

        p1 = np.mean(y)
        p0 = 1 - p1

        w1 = p0 / p1

        loss = nn.CrossEntropyLoss(weight=torch.tensor([1, w1], dtype=torch.float))
        optimizer = optim.Adam(self.parameters(), lr=lr)
        losses = []

        y = torch.tensor(y.to_numpy())

        y = y.to(torch.long)

        # w = 9 * torch.ones(len(y), requires_grad=False) * y + torch.ones(len(y), requires_grad=False)

        # print(w)

        for epoch in range(epochs):
            y_hat = self(X)
            loss_ = loss(y_hat, y)
            loss_.backward()

            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss_.item())
            print(f"Epoch {epoch} / {epochs} | Loss {np.mean(losses)}")

        return losses





