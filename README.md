# Diagnóstico Temprano de Parkinson mediante Machine Learning Integrando Datos Heterogéneos

## Contexto y Justificación

El proyecto se centra en la aplicación de técnicas de machine learning para el diagnóstico y seguimiento de la enfermedad de Parkinson mediante la integración de datos heterogéneos. Se utilizan datos estructurados provenientes de encuestas junto con series temporales multi-canal recogidas a partir de dispositivos wearables.

La detección temprana de Parkinson es fundamental, ya que permite iniciar tratamientos que pueden ralentizar la progresión de la enfermedad, mejorar la calidad de vida del paciente y reducir complicaciones asociadas. Diversas investigaciones han demostrado la importancia de prevenir, identificar y tratar el Parkinson debido a su impacto en la mortalidad y morbilidad de la población, especialmente en el contexto actual de envejecimiento poblacional y el creciente impacto de las enfermedades neurodegenerativas.

El objetivo principal es desarrollar modelos de clasificación multi-clase que permitan detectar, de forma no invasiva y en etapas tempranas, si una persona padece la enfermedad de Parkinson. Los principales desafíos abordados son:

- El desarrollo de modelos de clasificación multi-clase para la detección de pacientes con Parkinson.
- El entrenamiento de modelos robustos sobre datos que combinan características estructurales y series temporales de diferentes fuentes.
- El análisis y comparación de distintos modelos para evaluar la conveniencia de corregir el desbalance de clases en el set de datos.

Respecto al desbalance de clases, se exploran técnicas como el ajuste del umbral de decisión, modificación de los pesos de la función de pérdida y técnicas de resampling como SMOTE. Sin embargo, se discute críticamente su utilidad y se concluye que, en términos generales, es preferible no aplicar métodos de oversampling para corregir el desbalance.

Para el entrenamiento y validación de los modelos, se utilizó el [Parkinson’s Disease Smartwatch Dataset (PADS)](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/), que cuenta con datos de 469 individuos pertenecientes a tres grupos: pacientes con Parkinson confirmado, pacientes con diagnósticos diferenciales de parkinsonismo y sujetos sanos de control.

## Estructura del Proyecto

- **exploratory_data_analysis.ipynb**  
  Notebook dedicado al análisis exploratorio de datos (EDA), donde se exploran y visualizan las características principales del dataset, se identifican patrones y se realiza un analisis de los datos

- **classification.ipynb**  
  Notebook principal para el procesamiento de datos, entrenamiento y evaluación de modelos de clasificación. Incluye análisis de sobreajuste (overfitting) y análisis de importancia de características.  
  **Nota:** El entrenamiento completo de los modelos no se ejecuta en el notebook debido a su alto costo computacional. El código para el entrenamiento está presente, pero la ejecución se realiza externamente mediante un script.

- **train_models.py**  
  Script utilizado para el entrenamiento de los modelos en una instancia cloud. Permite aprovechar recursos computacionales avanzados y entrenar los modelos.

## Uso

1. **Análisis Exploratorio:**  
   Ejecutar `exploratory_data_analysis.ipynb` para explorar y comprender el dataset.

2. **Procesamiento y Evaluación:**  
   Utilizar `classification.ipynb` para preparar los datos, definir modelos, realizar validación cruzada, analizar el sobreajuste y la importancia de las variables.

3. **Entrenamiento Completo:**  
   Ejecutar `train_models.py` en una instancia cloud para entrenar los modelos con todos los datos y parámetros definidos.

## Consideraciones

- **Requisitos:**  
  - Python 3.12

Instala los requisitos con:
```bash
pip install -r requirements.txt