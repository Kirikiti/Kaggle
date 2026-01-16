import argparse
import pandas as pd

from preprocessing.pipeline import create_preprocessing_pipeline
from models.oof_generator import OOFGenerator


class Main:
    def __init__(self, train_path, target, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.target = target

    def run(self):
        print("Cargando datos...")

        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path) if self.test_path else None

        print("Creando pipeline de preprocesamiento...")
        pipe = create_preprocessing_pipeline()

        X_train = train_df.drop(columns=[self.target])
        y_train = train_df[self.target]

        print("Transformando train...")
        X_train_trans = pipe.fit_transform(X_train, y_train)

        if test_df is not None:
            print("Transformando test...")
            X_test_trans = pipe.transform(test_df)
        else:
            X_test_trans = None

        train_trans_df = pd.DataFrame(X_train_trans, columns=X_train.columns)
        train_trans_df[self.target] = y_train.values

        test_trans_df = (
            pd.DataFrame(X_test_trans, columns=X_train.columns)
            if X_test_trans is not None else None
        )

        print("Generando OOF...")
        oof_gen = OOFGenerator(
            train_df=train_trans_df,
            test_df=test_trans_df,
            target=self.target
        )

        oof_preds, test_preds, _ = oof_gen.fit_predict()

        print("Guardando oof_train.csv...")
        pd.DataFrame(oof_preds).to_csv("oof_train.csv", index=False)

        if test_preds is not None:
            print("Guardando pred_test.csv...")
            pd.DataFrame(test_preds).to_csv("pred_test.csv", index=False)

        print("Proceso completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--test", type=str, required=False)

    args = parser.parse_args()

    Main(
        train_path=args.train,
        target=args.target,
        test_path=args.test
    ).run()
