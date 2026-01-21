# 🧠 Reconocimiento de dígitos/letras manuscritas  
Guía rápida para [competir en Kaggle](https://www.kaggle.com/competitions/digit-recognizer/)

## 1. 📦 Preparación de los datos
- Convertir imágenes a escala de grises.
- Normalizar valores a rango [0, 1].
- Redimensionar (28×28 o 32×32 según el dataset).
- One-hot encoding de las etiquetas (A–Z o 0–9).
- Dividir en train/validación.

## 2. 🔧 Modelos recomendados

### 🌀 A. CNN básica (rápida y efectiva)
Funciona muy bien para letras y dígitos simples.

Arquitectura típica:
- Conv2D (32 filtros, 3×3) + ReLU  
- Conv2D (32 filtros, 3×3) + ReLU  
- MaxPooling  
- Dropout  
- Conv2D (64 filtros, 3×3)  
- MaxPooling  
- Flatten  
- Dense (128) + ReLU  
- Dense (n_clases) + Softmax

### 🚀 B. Modelos más potentes
- **ResNet18 / ResNet34**  
- **EfficientNet-B0**  
- **MobileNetV2** (rápido y ligero)

Estos suelen mejorar el score si el dataset es grande.

### 🧪 C. Ensembles
Combinar varios modelos (promedio de predicciones) suele mejorar el leaderboard.

---

## 3. 🎨 Data Augmentation
Muy útil para letras manuscritas:

- Rotaciones pequeñas (±10°)  
- Zoom  
- Shear  
- Shift horizontal/vertical  
- Pequeño ruido gaussiano  

Evita transformaciones que deformen demasiado la letra.

---

## 4. 🏁 Estrategia para competir
1. Entrenar una CNN básica para tener baseline.  
2. Probar modelos preentrenados (transfer learning).  
3. Ajustar augmentation.  
4. Hacer ensemble de los mejores modelos.  
5. Afinar el threshold o promediar logits para mejorar el score final.

---

## 5. 📊 Métricas
- **Accuracy** si es clasificación pura.  
- **F1-score** si las clases están desbalanceadas.  

---

## 6. 🧩 Código base (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

---

# 🔍 Modelos clásicos para reconocer dígitos/letras manuscritas  
*(sin usar redes neuronales)*

Estos algoritmos funcionan sorprendentemente bien en datasets como MNIST, EMNIST o competiciones de letras individuales.

---

## 🥇 1. SVM (Support Vector Machines)
- Uno de los mejores modelos clásicos para imágenes pequeñas.
- Con kernel RBF suele superar el 97–98% en MNIST.
- Requiere normalizar y a veces reducir dimensionalidad.

**Ventajas:** muy preciso, robusto.  
**Desventajas:** lento con muchos datos.

---

## 🥈 2. Random Forest
- Funciona bien sin mucha ingeniería.
- Captura relaciones no lineales.
- Rápido de entrenar.

**Ventajas:** fácil de usar, buen baseline.  
**Desventajas:** no llega al rendimiento de SVM.

---

## 🥉 3. Gradient Boosting / XGBoost / LightGBM
- Suelen superar a Random Forest.
- Muy buenos con features derivados de imágenes (HOG, PCA, etc.).

**Ventajas:** excelente rendimiento.  
**Desventajas:** requieren tuning.

---

## 🔢 4. KNN (k-Nearest Neighbors)
- Sorprendentemente fuerte en MNIST si se usa PCA o reducción de dimensión.
- Muy simple: no entrena, solo compara.

**Ventajas:** fácil, buen baseline.  
**Desventajas:** lento al predecir.

---

## 🧩 5. Logistic Regression (multiclase)
- Funciona mejor de lo que parece si las imágenes están bien normalizadas.
- Buen punto de partida.

**Ventajas:** rápido, interpretable.  
**Desventajas:** limitado para patrones complejos.

---

## 🧱 6. Naive Bayes
- Solo útil como baseline muy básico.
- Funciona mejor con datos binarizados.

---

## 🎨 7. HOG + Modelo clásico
Una combinación muy potente:
- Extraer **HOG (Histogram of Oriented Gradients)** de cada imagen.
- Entrenar un **SVM**, **Random Forest** o **XGBoost** encima.

Esto era el estándar antes de las CNN y sigue siendo competitivo.

---

## ⭐ Recomendación práctica
Para una competición de letras manuscritas:

1. **HOG + SVM (RBF)** → suele ser el mejor modelo clásico.  
2. **HOG + XGBoost** → muy competitivo.  
3. **PCA + SVM** → rápido y preciso.  
4. **KNN + PCA** → baseline sorprendentemente fuerte.

---

Si quieres, te preparo:
- un **notebook completo** con HOG + SVM,  
- una **comparativa de todos estos modelos**,  
- o una **pipeline optimizada** para Kaggle.

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
