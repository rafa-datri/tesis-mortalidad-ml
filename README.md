# Modelado de la Mortalidad con Machine Learning

Aplicación de Redes Neuronales Artificiales y XGBoost sobre una cartera de rentas vitalicias de una aseguradora chilena.

---

## Problema

Las tablas de mortalidad regulatorias, al construirse sobre poblaciones agregadas, pueden no reflejar con exactitud la experiencia real de carteras específicas. Esta desconexión afecta directamente el cálculo de primas, reservas matemáticas y la evaluación del riesgo biométrico en productos de largo plazo como las rentas vitalicias.

Este trabajo propone construir una tabla de mortalidad empírica ajustada a la experiencia de una cartera real de rentas vitalicias en Chile mediante técnicas de Machine Learning, y comparar sus resultados con la tabla regulatoria CB-H-2020.

---

## Metodología

Se implementa un pipeline unificado que compara dos modelos de clasificación binaria (probabilidad anual de fallecimiento):

- **Redes Neuronales Artificiales (MLP)** con Keras/TensorFlow.
- **XGBoost** (Extreme Gradient Boosting).

### Pipeline principal

1. **Preprocesamiento**: filtro por sexo (M/F/ALL), split train/valid/test, estandarización de variables.
2. **Balanceo de clases**: SMOTE (Synthetic Minority Over-sampling Technique) aplicado únicamente sobre el conjunto de entrenamiento.
3. **Entrenamiento**: tuning de hiperparámetros con RandomizedSearchCV / búsqueda dirigida.
4. **Calibración de probabilidades**: regresión isotónica sobre el conjunto de validación.
5. **Selección de umbral óptimo por bin de edad**: criterios COUNT_MATCH, F1-score o Recall.
6. **Agregación de qx**: cálculo de probabilidades por bin de edad usando persona-años (Ortega).
7. **Suavizado**: método de Whittaker-Henderson con selección de λ óptimo.
8. **Monotonía**: enforce de no-decrecimiento de qx con la edad.
9. **Comparación**: contra la tabla regulatoria CB-H-2020, con intervalos de confianza de Poisson (Delta method).

### Métricas de evaluación

Dado el fuerte desbalance de clases (fallecimientos como eventos raros), se emplean métricas apropiadas para clasificación desbalanceada:

- **F1-score** (media armónica entre precision y recall)
- **AUC** (área bajo la curva ROC)
- **MAE ponderado** (para comparación de tablas de mortalidad)
- **Precision** y **Recall**

---

## Requerimientos

- Python 3.9+
- Dependencias principales:
  - `numpy`, `pandas`, `scipy`
  - `scikit-learn`
  - `xgboost`
  - `tensorflow` / `keras`
  - `imbalanced-learn` (SMOTE)
  - `matplotlib`, `joblib`

Instalación rápida:

```bash
pip install numpy pandas scipy scikit-learn xgboost tensorflow imbalanced-learn matplotlib joblib
```

---

## Datos

Los datos utilizados corresponden a una cartera real de rentas vitalicias de una aseguradora chilena, provistos bajo acuerdo de confidencialidad. **Por este motivo, los datos no se incluyen en este repositorio.**

El pipeline espera dos archivos CSV en el directorio de trabajo:

- `DF_TOTAL_CB.csv`: base de asegurados con variables demográficas, temporales y flag de fallecimiento.
- `MORT_TABLE_CB_H_2020.csv`: tabla regulatoria de mortalidad CB-H-2020 (Chile) para comparación.

---

## Cómo ejecutar

1. Ajustar la ruta `MAIN_PATH` al inicio del script.
2. Colocar los archivos de datos en el directorio configurado.
3. Configurar flags según objetivo:
   - `RUN_XGB` / `RUN_NN`: seleccionar modelos a entrenar.
   - `SEX_FILTER`: "M", "F" o "ALL".
   - `OBJ_UMBRAL`: criterio de selección del umbral óptimo.
4. Ejecutar:

```bash
python NN_XGB_v3.py
```

Los resultados (modelos entrenados, gráficos, métricas y tablas) se guardan en `OUTPUT_MODEL_NN_XGB_final/`.

---

## Referencias principales

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
- Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*.
- Haykin, S. (2009). *Neural Networks and Learning Machines*.
- Ortega, A. (2004). *Tablas de mortalidad*. CELADE.
- Society of Actuaries (2015). *Experience Studies*.
- Lee, R. D., & Carter, L. R. (1992). *Modeling and forecasting U.S. mortality*.

---

## Contacto

Rafael D'Atri
📧 datri.rafa@gmail.com
🔗 [LinkedIn]([https://linkedin.com/in/rafaeldatri](https://www.linkedin.com/in/rafael-d-atri-7796b5204/)
