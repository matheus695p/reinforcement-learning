# Q-Trader

Una implementación de Q-learning aplicado a la negociación de acciones (a corto plazo). El modelo utiliza ventanas de precios de cierre de n días para determinar si la mejor acción a tomar en un momento dado es comprar, vender o sentarse.


### Resultados

Se entrena el modelo con datos GSPC de 2010 y lo probamos con el primer trimestre de 2011.
S&P 500, 2011Q1. Ganancia de $92.84:


## SETUP

```
mkdir models
python train.py GSPC_10 5 30
```

Luego, cuando finalice el entrenamiento, puede evaluar con el conjunto de datos de prueba:

```
python evaluate.py GSPC_2011-03 model_ep30
```
