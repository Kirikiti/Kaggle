# ✅ 1. Activa tu entorno conda
```bash
conda activate mi_entorno
```
Si aún no está creado, se crea así:
```bash
conda create -n mi_entorno python=3.10
```
# ✅ 2. Clonas el Repo en tu máquina local
```bash
git clone https://github.com/Kirikiti/Kaggle.git
```
```bash
cd Kaggle
```
Así se instala Git en entorno conda si aún no está instalado:
```bash
conda install git
```
# ✅ 3. Instalas los requirements (librerias necesarias)
```bash
pip install -r requirements_oof.txt
```
# 🚀 2. Cómo ejecutarlo
## Solo train (solo genera OOF)
```bash
python -m src.scripts.run_oof --train <ruta_al_csv_de_entrenamiento> --target <Nombre Variable dependiente>
```
Ejemplos válidos:
```bash
python src/scripts/run_oof.py --train data/mi_train.csv --target Precio
python src/scripts/run_oof.py --train ./datasets/train_2024_v3.csv --target Precio
python src/scripts/run_oof.py --train https://miweb.com/datos/train.csv --target Precio
```
Esto generará:
```bash
oof_train.csv
```
## Train + Test (genera OOF + predicciones)

```bash
python -m src.scripts.run_oof --train <ruta_al_csv_de_entrenamiento> --target <Nombre Variable dependiente> --test <ruta_al_csv_de_entrenamiento>
```
Esto generará:
```bash
oof_train.csv
pred_test.csv
```
Los CSV siempre se guardan en tu máquina local, en el directorio desde el que ejecutas el script.
GitHub no interviene en la ejecución; solo sirve como repositorio del código.
