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

## Estructura del repositorio
