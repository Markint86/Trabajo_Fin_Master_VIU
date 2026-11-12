# Resumen detallado del TFM

Este documento resume el contenido de la memoria del Trabajo Fin de Máster, que no se publica en el repositorio. Amplía lo que cuenta el [README](../README.md) con el contexto, las decisiones tomadas y la discusión de resultados de cada fase.

**Título:** Detección de pedidos con alta probabilidad de devolución y aplicación de medidas preventivas
**Máster:** Máster Universitario en Inteligencia Artificial, Universidad Internacional de Valencia (VIU), 2025–2026

---

## 1. Contexto y motivación

- En el comercio electrónico, las tasas de devolución rondan el **20 %**, frente a menos del 10 % en tienda física. La moda encabeza el ranking, con alrededor de un **25 %**.
- Las devoluciones encarecen la logística, complican el inventario y tienen un coste medioambiental por la logística inversa.
- Entre los comportamientos habituales que las provocan están el *bracketing* (comprar varias tallas o colores para quedarse con una), el *wardrobing* (usar el producto y devolverlo) y problemas de talla o ajuste, que suponen alrededor del 65 % de las causas declaradas.
- **Idea del trabajo:** si se detecta a tiempo qué pedidos tienen más riesgo de devolverse, se pueden aplicar **acciones preventivas** (por ejemplo, un email con una guía de tallas o una confirmación del pedido) antes de que la devolución ocurra.

## 2. Estado del arte y aportación

| Trabajo | Aportación | Lo que no cubre |
|---------|-----------|-----------------|
| Cui et al. (2020) | Comparación rigurosa de modelos con datos reales (LASSO, el mejor) | Interpretabilidad y acciones preventivas |
| Karl (2024) | Revisión sistemática de 25 publicaciones: los métodos de árboles, boosting y redes neuronales suelen ser los mejores | No desarrolla modelos propios |
| Mishra y Dutta (2024) | Predicción combinada con la identificación de variables influyentes | Acciones preventivas |
| Duong et al. (2024) | Modelos interpretables a partir de reseñas de clientes | Depende de información posterior a la compra |

**Aportación de este TFM:** une en un mismo flujo la **predicción** (comparando 5 modelos), la **explicabilidad con SHAP** y una **estimación del impacto económico** de aplicar acciones preventivas.

## 3. Datos y preprocesado ([notebook 01](../notebooks/TFM_01_EDA_Preprocesado.ipynb))

**Dataset:** [E-commerce Sales Transactions Dataset](https://www.kaggle.com/datasets/miadul/e-commerce-sales-transactions-dataset) (Kaggle, sintético), con 34.500 pedidos y 17 columnas. No tiene valores nulos ni duplicados.

**Principales hallazgos del EDA:**
- **Desbalanceo:** solo el 5,52 % de los pedidos se devuelve. Un modelo que dijera siempre "no devuelto" acertaría el 94,48 %, así que la accuracy no sirve como métrica.
- **Asimetría:** `total_amount` (8,22), `profit_margin` (6,04) y `price` (3,70) tienen una cola larga a la derecha.
- **Atípicos (regla 1,5·IQR):** son el 10,99 % en `total_amount`, el 10,53 % en `price` y el 8,77 % en `profit_margin`. **No se eliminan**, porque son pedidos reales de alto importe y quitarlos sesgaría la estimación económica.
- **Relación con la variable objetivo:** las correlaciones son débiles. Los pedidos devueltos tienen de media un importe algo mayor (204 USD frente a 168 USD) y un coste de envío algo mayor (6,63 USD frente a 6,12 USD). Entre las categóricas, solo `category` se aleja claramente de la tasa media: **Fashion** devuelve más y **Grocery** menos.
- **Multicolinealidad:** `total_amount` correlaciona 0,80 con `price` y 0,89 con `profit_margin`.
- **`order_date`:** agrupada por mes o por día de la semana no muestra ningún patrón estacional, así que se descarta.

**Pasos de preprocesado (en este orden, para evitar fuga de información):**
1. Se eliminan `order_id`, `customer_id`, `product_id` y `order_date`.
2. Se binariza la variable objetivo `returned` (No = 0, Yes = 1).
3. Split **estratificado 80/20** (27.600 / 6.900 pedidos) **antes** de cualquier transformación.
4. One-Hot Encoding de las categóricas (`drop="first"`).
5. `StandardScaler` en las numéricas, ajustado solo con train.

Resultado: **25 variables** de entrada.

## 4. Modelado ([notebook 02](../notebooks/TFM_02_Modelado.ipynb))

**Metodología común:**
- `StratifiedKFold` con 5 particiones (3 para la MLP) y búsqueda de hiperparámetros con `GridSearchCV` o `RandomizedSearchCV`.
- Métrica de selección: **PR-AUC**, la más sensible a la clase minoritaria. También se reportan Recall (no dejar escapar devoluciones), F2 (da más peso al Recall), F1 y ROC-AUC.
- Desbalanceo tratado con `class_weight="balanced"` en los modelos de scikit-learn y Keras, y con `scale_pos_weight` en XGBoost.

**Comportamiento de cada modelo sobre el conjunto de test** (381 devoluciones reales):

- **Regresión Logística** (`C=0.01`, L1): detecta **259 de 381** devoluciones (68 %), a cambio de 3.257 falsas alarmas. La penalización L1 anula los coeficientes poco útiles; los más influyentes son la categoría (Grocery, Fashion) y `shipping_cost`.
- **Árbol de Decisión** (entropía, profundidad 5): resultados muy parecidos a la Regresión Logística. El nodo raíz es `category_Grocery`, y le siguen `category_Fashion` y `shipping_cost`.
- **Random Forest** (200 árboles, profundidad 10): es el que mejor acierta los pedidos no devueltos, pero **solo detecta 136 de 381** devoluciones. Su importancia nativa (reducción de impureza) prioriza las variables económicas.
- **XGBoost** (300 árboles, `learning_rate=0.01`): tiene el mejor PR-AUC (0,0789), aunque su Recall es menor que el de la Regresión Logística y el Árbol.
- **MLP** (1 capa oculta de 32 neuronas, sin dropout): ganó la arquitectura más sencilla de las probadas, señal de que el problema no exige mucha capacidad. La curva de pérdida no llegó a estabilizarse, así que más épocas podrían haberla mejorado.

**Conclusión:** todos los modelos mejoran el *baseline* aleatorio (PR-AUC 0,0552) **entre 1,3 y 1,43 veces**, con diferencias pequeñas entre ellos.

## 5. Interpretabilidad SHAP ([notebook 03](../notebooks/TFM_03_SHAP.ipynb))

| Modelo | Explicador | Escala |
|--------|-----------|--------|
| Regresión Logística | `LinearExplainer` (exacto) | log-odds |
| Árbol, Random Forest | `TreeExplainer` (exacto) | probabilidad |
| XGBoost | `TreeExplainer` (exacto) | log-odds |
| MLP | `DeepExplainer` (aproximado) | probabilidad |

Los valores SHAP se calculan sobre los 6.900 pedidos de test. La importancia de cada variable es la media de |SHAP|, expresada como porcentaje del total de cada modelo para poder comparar entre modelos.

**Hallazgos:**
- **Las 3 variables más importantes son las mismas en todos los modelos:** `category_Grocery` (reduce el riesgo de devolución), `shipping_cost` (a mayor coste de envío, más riesgo) y `category_Fashion` (aumenta claramente el riesgo).
- Lo que cambia entre modelos es **cuánto se concentra** la importancia: el 86 % en las 3 primeras en la Regresión Logística, frente al 38 % en la MLP, que reparte peso también entre variables secundarias.
- **SHAP corrige las importancias nativas de los modelos.** En Random Forest, `category_Grocery` aparecía en sexto lugar por reducción de impureza y SHAP la sitúa en primer lugar. En XGBoost, `shipping_cost` sube del octavo al segundo puesto. Las variables continuas se usan como punto de corte en muchos nodos sin aportar tanto a la predicción final, mientras que las binarias separan de forma más nítida.
- Que los modelos complejos no encuentren señal adicional encaja con el EDA: **no hay relaciones no lineales relevantes que explotar**.

## 6. Selección del modelo ([notebook 04](../notebooks/TFM_04_SeleccionModelo.ipynb))

Se combinan tres criterios: rendimiento, coste computacional e interpretabilidad. Ninguno basta por separado.

- **Rendimiento:** la diferencia de PR-AUC entre XGBoost y la Regresión Logística es de solo **0,0006**, así que no permite decidir. El Recall sí desempata: la Regresión Logística tiene el más alto (0,68).
- **Coste computacional:** la Regresión Logística se entrena en 5,4 s, frente a los 85,2 s de XGBoost y los 306,6 s de Random Forest.
- **Interpretabilidad:** la Regresión Logística tiene la mayor concentración SHAP (86 %), lo que la hace más fácil de explicar a negocio.

**Modelo elegido: Regresión Logística.** Un modelo simple e interpretable iguala o supera a los complejos en este problema.

## 7. Impacto económico ([notebook 05](../notebooks/TFM_05_ImpactoEconomico.ipynb))

**Modelo de coste:**

```
Beneficio neto(c) = 0,30 × Σ total_amount(VP)  −  c × (N_VP + N_FP)
                    └──── ahorro bruto ────┘     └ coste de las acciones ┘
```

- **Ahorro bruto:** se toma el 30 % del importe de las devoluciones detectadas a tiempo (VP) como coste evitado de gestionar la devolución, según SmartRoutes (2026).
- **Coste de las acciones:** la acción preventiva, con coste `c`, se aplica a **todos** los pedidos marcados (VP + FP), porque se aplica antes de saber si el pedido se iba a devolver.

**Resultados sobre test:**
- Ahorro bruto: **22.240,71 USD**.
- Pedidos marcados: 3.516.
- **Coste de equilibrio: 6,33 USD por acción.** Por debajo de ese coste, el sistema es rentable.
- Oportunidad perdida por las devoluciones no detectadas (FN): 3.927,38 USD.

El resultado escala linealmente con el porcentaje de ahorro supuesto. La fuente habla de un ahorro de "hasta un 30 %", así que con un porcentaje menor el coste de equilibrio bajaría en la misma proporción. Además, las cifras corresponden a 6.900 pedidos de test y habría que extrapolarlas al volumen real de negocio.

## 8. Conclusiones

1. Se ha construido un sistema completo de **predicción + explicabilidad + estimación económica** para anticipar devoluciones.
2. La señal predictiva del dataset es **limitada**: el mejor PR-AUC es 0,0789 frente a 0,0552 del baseline.
3. El hallazgo principal no es qué modelo predice mejor en términos absolutos, sino que **un modelo simple e interpretable iguala a los complejos** y que **tres variables concentran la señal** de forma consistente en todos ellos.
4. El sistema es económicamente viable si cada acción preventiva cuesta **menos de 6,33 USD** por pedido marcado.

## 9. Limitaciones y trabajo futuro

- **Dataset sintético:** no hay validación con datos reales de una empresa.
- **Sin historial del cliente:** faltan variables como las devoluciones previas, que en la literatura son muy predictivas.
- **Modelo económico simplificado:** el ahorro del 30 % es fijo y el coste `c` es el mismo para todos los pedidos, sin diferenciar por categoría ni por nivel de riesgo.
- **MLP sin estabilizar** por limitaciones de tiempo y cómputo.
- **Próximos pasos:** validar con datos reales que incluyan comportamiento del cliente y del producto, usar costes de acción diferenciados por segmento de riesgo y ajustar el umbral de decisión según el coste.

## Referencias principales

- Cui, H., Rajagopalan, S. y Ward, A. R. (2020). Predicting product return volume using machine learning methods. *European Journal of Operational Research*, 281(3), 612–627.
- Duong, Q. H., Zhou, L., Van Nguyen, T. y Meng, M. (2024). Understanding and predicting online product return behavior: an interpretable machine learning approach. *International Journal of Production Economics*, 280, 109499.
- Karl, D. (2024). Forecasting e-commerce consumer returns: a systematic literature review. *Management Review Quarterly*, 75(3), 1–56.
- Mishra, A. y Dutta, P. (2024). Return management in e-commerce firms: A machine learning approach to predict product returns and examine variables influencing returns. *Journal of Cleaner Production*, 477, 143802.
- Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*, 4765–4774.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Marriott, J., Bektaş, T., Leung, E. K. H. y Lyons, A. (2025). The billion-pound question in fashion e-commerce: Investigating the anatomy of returns. *Transportation Research Part E*, 194, 103904.
- SmartRoutes (2026). Returns and reverse logistics: Key data for retail and delivery teams.
- Eightx (2026). Average ecommerce return rate 2026.
