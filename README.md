import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import joblib
import os


def train_models(mydf: pd.DataFrame):

    currentpath = os.path.dirname(os.path.abspath(__file__))

    independentcols = [
        'renda',
        'idade',
        'etnia',
        'sexo',
        'casapropria',
        'outrasrendas',
        'estadocivil',
        'escolaridade'
    ]

    X = mydf[independentcols]
    y = mydf['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODELO 1
    independentcols_m1 = independentcols.copy()

    clf = rfc(
        n_estimators=20,
        max_depth=None,
        random_state=42
    )

    clf.fit(
        X=X_train[independentcols_m1],
        y=y_train
    )

    clf.independentcols = independentcols_m1

    clf_acuracia = clf.score(
        X=X_test[independentcols_m1],
        y=y_test
    )

    print(f"Modelo 01 criado com acurácia: {clf_acuracia}")


    # MODELO 2
    independentcols_m2 = independentcols.copy()

    independentcols_m2.remove('etnia')

    rgs = rfr(
        n_estimators=20,
        max_depth=None,
        random_state=42
    )

    rgs.fit(
        X=X_train[independentcols_m2],
        y=y_train
    )

    rgs.independentcols = independentcols_m2

    rgs_acuracia = rgs.score(
        X=X_test[independentcols_m2],
        y=y_test
    )

    print(f"Modelo 02 criado com acurácia: {rgs_acuracia}")


    # salva modelos
    os.makedirs(f'{currentpath}/models', exist_ok=True)

    joblib.dump(
        clf,
        f'{currentpath}/models/modelo01.joblib'
    )

    joblib.dump(
        rgs,
        f'{currentpath}/models/modelo02.joblib'
    )

    print("Modelos atualizados com sucesso")


if __name__ == "__main__":

    mydf = pd.read_csv('./datasets/BaseDefault01.csv')

    train_models(mydf=mydf)
