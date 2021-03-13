import numpy as np
import pandas as pd


class QLearningTable:
    """
    Algoritmo de Q-learning. Es el cerebro detrás del agente, es donde se
    implementa la ecuación de bellman.

    """

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9,
                 e_greedy=0.1):
        """
        Constructor, los atributos necesarios para hacer para el obtejecto
        Q-table
        Parameters
        ----------
        actions : int
            [0, 1, 2, 3].
        learning_rate : TYPE, optional
            DESCRIPTION. The default is 0.01.
        reward_decay : TYPE, optional
            DESCRIPTION. The default is 0.9.
        e_greedy : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        None.

        """
        # número de posibles acciones
        self.actions = actions
        # learning rate
        self.lr = learning_rate
        # bajada de los rewards
        self.gamma = reward_decay
        # exporar vs explotar parámetro
        self.epsilon = e_greedy
        # Q-Table de acciones
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        """
        Escoger la acción que se hace en función de la observación hecha
        en estado actual y si estamos en etapa de explotación o de exploración
        Parameters
        ----------
        observation : TYPE
            DESCRIPTION.
        Returns
        -------
        action : int [arriba, abajo, derecha, izquierda]
            [0, 1, 2, 3].
        """
        # agregar observación a la tabla
        self.add_state(observation)

        # selección de la acción en etapa de exploración
        if np.random.uniform() < self.epsilon:
            # escoger acción aleatoria
            action = np.random.choice(self.actions)
        else:
            # estamos en etapa de explotación por lo tanto:
            # 1) encontrar los registros de observación actual
            # 2) reindex la data resultado
            # 3) devolver la acción con el valor más alto
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(
                state_action.index))
            action = state_action.idxmax()
        return action

    def learn(self, s, a, r, s_):
        """
        Método de aprendizaje según ecuación de bellman
        Parameters
        ----------
        s : estado actual
            DESCRIPTION.
        a : string
            DESCRIPTION.
        r : int
            DESCRIPTION.
        s_ : int
            DESCRIPTION.

        Returns
        -------

        """
        # agreagar la observación siguiente en la tabla (s_)
        self.add_state(s_)
        # Elija el mejor q-value para el par (s, a); Q(s, a)
        # buscar en los registros de la acción
        q_predict = self.q_table.loc[s, a]

        # si aún no hemos perdido el juego
        if s_ != 'terminal':
            # ToDo: approximate the expected future reward based on Bellman equation:
            # Q'() = r + gamma * [max_a' Q(s',a')]
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()

        # el siguiente estado es terminal
        else:
            q_target = r

        # actualizar q-value en la tabla según ecuación de bellman
        # Q(s, a) = Q(s, a) + learning_rate [r + gamma max_a' Q(s', a') - Q(s, a)]
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def add_state(self, state):
        """
        Agregar el nuevo estado en la tabla
        Parameters
        ----------
        state : terminal or not
            si aún no ha perdido buscando el agente.
        Returns
        -------
            Agrega el estado a la tabla
        """
        if state not in self.q_table.index:
            # agregar nuevo estado en la tabla en el caso de que no este
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions),
                          index=self.q_table.columns,
                          name=state))
