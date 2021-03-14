import numpy as np
import pandas as pd


class Market:
    """
    Crear el entorno de entrenamiento como una lista de la historia de precios
    """

    def __init__(self, window_size, stock_name):
        """
        Constructor del enviroment del agente
        Parameters
        ----------
        window_size : int
            tamaño de ventana temporal hacia atrás en la data.
        stock_name : string
            nombre del archivo.
        """
        # traer el archivo como texto plano
        self.data = self.__get_stock_data(stock_name)
        # definir el estado en cúal estamos
        self.states = self.__get_all_window_prices_diff(self.data, window_size)
        # indice último
        self.index = -1
        # traer esa última data
        self.last_data_index = len(self.data) - 1

    def __get_stock_data(self, key):
        """
        Leer el csv y separarlo en 4 columnas
        Parameters
        ----------
        key : string
            nombre del archivo.
        Returns
        -------
        vec : list
            precio de cerrada del activo.
        """
        vec = []
        lines = pd.read_csv("data/" + key + ".csv")
        lines["Close"] = lines["Close"].apply(float)
        vec = lines["Close"].to_list()
        return vec

    def __get_window(self, data, t, n):
        """
        Sacar la ventana temporal de data con un largo

        Parameters
        ----------
        data : list
            lista de precios.
        t : int
            largo total de la lista.
        n : int
            mirar hacia atrás.
        """

        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * \
            [data[0]] + data[0:t + 1]
        res = []
        for i in range(n - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])

    def __get_all_window_prices_diff(self, data, n):
        """
        Procesar previamente los datos para crear una lista de estados de
        tamaño de ventana
        Parameters
        ----------
        data : list
            DESCRIPTION.
        n : int
            largo de la ventana.
        Returns
        -------
        processed_data : list
            lista con los estados en el tamaño de la ventana.
        """
        processed_data = []
        for t in range(len(data)):
            state = self.__get_window(data, t, n + 1)
            processed_data.append(state)
        return processed_data

    def reset(self):
        """
        Resetear el últiumo indice
        """
        self.index = -1
        return self.states[0], self.data[0]

    def get_next_state_reward(self, action, bought_price=None):
        """
        Obtener la recompenza del estado siguiente, la idea del siguiente
        estado es que pueda avanzar por el vector de precios, como el
        siguiente estado
        Parameters
        ----------
        action : int
            vender, quedarse o comprar.
        bought_price : TYPE, optional
            DESCRIPTION. The default is None.
        Returns
        -------
        next_state : TYPE
            DESCRIPTION.
        next_price_data : TYPE
            DESCRIPTION.
        reward : TYPE
            DESCRIPTION.
        done : TYPE
            DESCRIPTION.

        """
        self.index += 1
        # cuando se vuelva a empezar setar el indice a cero
        if self.index > self.last_data_index:
            self.index = 0
        # el estado siguiente es sumar uno al indice
        next_state = self.states[self.index + 1]
        # el precio siguiente es el mismo
        next_price_data = self.data[self.index + 1]
        price_data = self.data[self.index]

        # recompenza a cero
        reward = 0
        # si la acción es 2 y el precio de compra no es el default
        if action == 2 and bought_price is not None:
            reward = max(price_data - bought_price, 0)
        # solo termina el episodio cuando paso todo el vector
        done = True if self.index == self.last_data_index - 1 else False
        return next_state, next_price_data, reward, done
