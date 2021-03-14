import warnings
import random
import numpy as np
from collections import deque
from keras.models import (Sequential, load_model)
from keras.layers import Dense
# from keras.layers impot (BatchNormalization, Dropout)
from keras.optimizers import Adam
warnings.filterwarnings("ignore")


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        """
        Cosntructor de la clase agente
        Parameters
        ----------
        state_size : TYPE
            DESCRIPTION.
        is_eval : boolean, optional
            definir si empezamos el modelo de cero o no. The default is False.
        model_name : string, optional
            DESCRIPTION. nombre del modelo en el caso que se quiera utilizar
            "".
        """
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []
        # dias anteriores normalizados
        self.state_size = state_size
        # tamaño del espacio de eacciones (quedarse, vender, comprar)
        self.action_size = 3
        # no quemar la memorio con un largo máximo
        self.memory = deque(maxlen=1000)
        # nombre del modelo que esta entrenado en el caso de que lo este
        self.model_name = model_name
        self.is_eval = is_eval

        # parámetros de la ecuación de bellman
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # cargar el modelo si estamos en evaluación, del caso contrario
        # empezar de cero
        self.model =\
            load_model(
                "models/" + model_name) if is_eval else self.create_model()

    def create_model(self):
        """
        Crear estructura del cerebro del agente, deep learning model
        Returns
        -------
        model : objt
            modelo predictivo de recompenzas.
        """
        model = Sequential()
        model.add(Dense(units=256,
                        input_dim=self.state_size,
                        activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        model.add(Dense(units=128, activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        # model.add(Dense(units=32, activation="relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))
        model.add(Dense(self.action_size,
                        activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def reset(self):
        """
        Volver todo a cero
        """
        self.__inventory = []
        self.__total_profit = 0
        self.action_history = []

    def act(self, state, price_data):
        """
        implementar la acción
        Parameters
        ----------
        state : int
            indice.
        price_data : float
            precio.
        Returns
        -------
        action : TYPE
            DESCRIPTION.
        bought_price : TYPE
            DESCRIPTION.
        """
        if not self.is_eval and np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            #  Predecir cuál sería la acción posible para un estado dado
            # predecir q-value del estado actual
            options = self.model.predict(state)

            # elige la acción con mayor probabilidad
            # seleccionar el q-value con el valor más alto posible
            action = np.argmax(options[0])
        # en el caso de que no haya nada
        bought_price = None
        # no hacer nada
        if action == 0:
            print(".", end='', flush=True)
            self.action_history.append(action)
        # comprar
        elif action == 1:
            self.buy(price_data)
            self.action_history.append(action)
        # vender
        elif action == 2 and self.has_inventory():
            bought_price = self.sell(price_data)
            self.action_history.append(action)
        # acción es 2 (vender) pero no tenemos nada en inventario para vender.
        else:
            self.action_history.append(0)

        return action, bought_price

    def experience_replay(self, batch_size):
        """

        Parameters
        ----------
        batch_size : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        mini_batch = []
        large = len(self.memory)
        # agregar solo los últimos 1000 valores dependiendo de la memoria
        for i in range(large - batch_size + 1, large):
            mini_batch.append(self.memory[i])
        # iterar
        for state, action, reward, next_state, done in mini_batch:
            if done:
                target = reward
            else:
                # actualizar q_value = reward + gamma * [max_a' Q(s',a')]
                # esto es Q(s', a') para todos los posibles a'
                next_q_values = self.model.predict(next_state)[0]
                # actualizar objectivo q_value usando la ecuación de Bellman
                target = reward + self.gamma * np.amax(next_q_values)

            # predecir q_value para el estado actual
            predicted_target = self.model.predict(state)
            # actualizar acciones
            # sustituir q_value objetivo por el valor predicho
            predicted_target[0][action] = target
            # Entrenar el modelo con valores de acción actualizados
            # y el con nuevo q_value
            self.model.fit(state, predicted_target, epochs=1, verbose=0)

        #  Hacer epsilon más pequeño con el tiempo, por lo tanto,
        #  hacer más explotación que exploración
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def buy(self, price_data):
        """
        Método de compra, en este caso solo se agregar al inventario el precio
        Parameters
        ----------
        price_data : float
            fijar el precio de compra.
        """
        self.__inventory.append(price_data)
        print("Comprar: {0}".format(self.format_price(price_data)))

    def sell(self, price_data):
        """
        Vender, para ello sacamos el precio del inventario
        Parameters
        ----------
        price_data : TYPE
            DESCRIPTION.
        Returns
        -------
        bought_price : TYPE
            DESCRIPTION.

        """
        # sacar el último del inventario
        bought_price = self.__inventory.pop(0)
        # a lo que compre menos lo que vendi
        self.__total_profit += price_data - bought_price
        print(
            "Vender: {0} | Ganancia: {1}".format(
                self.format_price(price_data),
                self.format_price(price_data - bought_price)))
        return bought_price

    def get_total_profit(self):
        """
        Sacar sumado de lo ganado hasta el momento
        """
        return self.format_price(self.__total_profit)

    def has_inventory(self):
        """
        Si tengo o no inventario
        """
        return len(self.__inventory) > 0

    def format_price(self, n):
        """
        Retornar el precio del inventario
        """
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
