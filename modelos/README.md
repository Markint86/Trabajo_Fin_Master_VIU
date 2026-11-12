# Modelos entrenados

Modelos con los mejores hiperparámetros encontrados en `notebooks/TFM_02_Modelado.ipynb`. Todos esperan como entrada las **25 variables preprocesadas** generadas por el notebook 01, en el mismo orden de columnas.

| Fichero | Modelo | Hiperparámetros | PR-AUC test |
|---------|--------|-----------------|:---:|
| `logistic_regression.joblib` | Regresión Logística (**modelo seleccionado**) | `C=0.01`, `penalty="l1"`, `solver="liblinear"`, `class_weight="balanced"` | 0,0783 |
| `decision_tree.joblib` | Árbol de Decisión | `criterion="entropy"`, `max_depth=5`, `min_samples_leaf=10`, `class_weight="balanced"` | 0,0719 |
| `random_forest.joblib` | Random Forest | `n_estimators=200`, `max_depth=10`, `min_samples_leaf=5`, `max_features="sqrt"`, `class_weight="balanced"` | 0,0724 |
| `xgboost_model.joblib` | XGBoost | `n_estimators=300`, `max_depth=6`, `learning_rate=0.01`, `min_child_weight=5`, `gamma=0.1`, `colsample_bytree=0.6`, `scale_pos_weight` | 0,0789 |
| `mlp_keras.keras` | MLP (Keras) | 25 → Dense(32, ReLU) → Dense(1, sigmoid), Adam `lr=0.001` | 0,0726 |

## Carga

```python
import joblib
from tensorflow import keras

lr  = joblib.load("modelos/logistic_regression.joblib")
xgb = joblib.load("modelos/xgboost_model.joblib")
mlp = keras.models.load_model("modelos/mlp_keras.keras")
```

Los `.joblib` se serializaron con scikit-learn 1.5.2. Con versiones posteriores se cargan correctamente, aunque muestran un `InconsistentVersionWarning`.
