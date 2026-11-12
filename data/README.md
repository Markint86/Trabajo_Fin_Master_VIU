# Datos

Esta carpeta no se versiona (ver `.gitignore`), salvo este fichero.

## Dataset original

- **Nombre:** [E-commerce Sales Transactions Dataset](https://www.kaggle.com/datasets/miadul/e-commerce-sales-transactions-dataset) (Kaggle)
- **Fichero:** `ecommerce_sales_34500.csv`
- **Tamaño:** 34.500 pedidos, 16 variables + variable objetivo `returned` (Yes / No)

Descarga el CSV desde el enlace anterior y guárdalo como:

```
data/ecommerce_sales_34500.csv
```

## Ficheros generados

Al ejecutar `notebooks/TFM_01_EDA_Preprocesado.ipynb` se generan aquí los conjuntos ya preprocesados (One-Hot Encoding + StandardScaler, split estratificado 80/20 con `random_state=42`):

| Fichero | Contenido |
|---------|-----------|
| `X_train_proc.csv` | 27.600 × 25 variables de entrenamiento |
| `X_test_proc.csv` | 6.900 × 25 variables de test |
| `y_train.csv` | Variable objetivo de entrenamiento (0/1) |
| `y_test.csv` | Variable objetivo de test (0/1) |

Los notebooks 02 y 03 leen estos ficheros. El notebook 05 vuelve a hacer el split a partir del CSV original para recuperar el `total_amount` sin escalar de cada pedido de test.
