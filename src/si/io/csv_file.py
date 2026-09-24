import pandas as pd


def read_csv(filename, sep=",", features=True, label=True):
    df = pd.read_csv(
        filename,
        sep=sep,
        header=0 if features else None
    )

    if label:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df
        y = None

    feature_names = list(X.columns) if features else None

    return Dataset(X, y, feature_names)


def write_csv(filename, dataset, sep=",", features=True, label=True):
    data = dataset.X.copy()

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if features and dataset.features is not None:
        data.columns = dataset.features

    if label and dataset.y is not None:
        data["y"] = dataset.y

    data.to_csv(
        filename,
        sep=sep,
        index=False,
        header=features
    )