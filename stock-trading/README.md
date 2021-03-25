# trader

Una implementación de Q-learning aplicado a la negociación de acciones (a corto plazo). El modelo utiliza ventanas de precios de cierre de n días para determinar si la mejor acción a tomar en un momento dado es comprar, vender o sentarse, espacio de discreto de acciones.


### resultados

Se entrena el modelo con datos GSPC de 2010 y lo probamos con el primer trimestre de 2011.
S&P 500, 2011Q1.
Ganancia de $91.02:


## SETUP

```
mkdir models
python train.py 

```

Luego, cuando finalice el entrenamiento, puede evaluar con el conjunto de datos de prueba:

```
python evaluate.py 
```
