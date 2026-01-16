# ✅ 1. Activa tu entorno conda
```bash
conda activate mi_entorno
```
# ✅ 2. Clonas el Repo en tu máquina local
```bash
git clone https://github.com/Kirikiti/Kaggle.git
```
```bash
cd Kaggle
```
# ✅ 3. Instalas los requirements (librerias necesarias)
```bash
pip install -r requirements_oof.txt
```
# 🚀 2. Cómo ejecutarlo
## Solo train (solo genera OOF)
```bash
python src/scripts/run_oof.py --train train.csv --target Precio
```
Esto generará:
```bash
oof_train.csv
```
## Train + Test (genera OOF + predicciones)

```bash
python src/scripts/run_oof.py --train train.csv --target Precio --test test.csv
```
Esto generará:
```bash
oof_train.csv
pred_test.csv
```
Los CSV siempre se guardan en tu máquina local, en el directorio desde el que ejecutas el script.
GitHub no interviene en la ejecución; solo sirve como repositorio del código.
