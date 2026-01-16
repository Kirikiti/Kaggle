import pandas as pd

class DataLoader:
    """
    Clase para cargar train y test desde rutas locales o URLs.
    """
    def __init__(self, train_path, test_path=None):
        self.train_path = train_path
        self.test_path = test_path

    def load_train(self):
        print(f"Cargando train desde: {self.train_path}")
        return pd.read_csv(self.train_path)

    def load_test(self):
        if self.test_path is None:
            print("No se proporcionó test_path, se omite test.")
            return None
        print(f"Cargando test desde: {self.test_path}")
        return pd.read_csv(self.test_path)
