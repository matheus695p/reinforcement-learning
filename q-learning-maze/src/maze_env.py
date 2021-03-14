import sys
import time
import numpy as np
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

# Parámetros del entorno
# pixeles por cuadrado
UNIT = 40

# El minimo es 10 en ambos
# alto
MAZE_H = 10
# ancho
MAZE_W = 10


class MazeEnviroment():
    """
    El objetivo de esta clase es construir el entorno de etranamiento del
    juego donde habrá la grilla de (clásico buscaminas) y 3 grillas que repre-
    sentarán el infierno (2 celdas), tierra (el resto) y oro (1 celda) y el
    agente que estará moviendose

    El entorno de Maze:
        Cuadricula de mosaicos.
    Rectangulo rojo:          El agente explorando.
    Rectangulo negro:        el infierno un hoyo negro      [reward = -1].
    circulo amarillo:        encontro orito                 [reward = +1].
    Todos los demás estados: tierra esta explorando         [reward = 0].

    """

    def __init__(self):
        """
        Constructor de la clase que servira como entorno de entrenamiento
        del agente
        Returns
        -------
        Parámetros de entrada.
        """
        self.window = tk.Tk()
        self.window.title('Afirmate CTM !! con Q-Learning')
        self.window.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        # posibles acciones que no necesita conover por el momento pero son
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.build_grid()

    def build_grid(self):
        """
        construir los objetos necesarios
        """
        self.canvas = tk.Canvas(self.window, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # grillar un tabla de UNIT * UNIT, iterar para ambos lados
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # punto de comienzo en el centro
        origin = np.array([int(UNIT / 2), int(UNIT / 2)])

        # definir centro de la celda
        hell1_center = origin + np.array([UNIT * 2, UNIT])

        # infierno 1
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # infierno 2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
       # infierno 3
        hell3_center = origin + np.array([UNIT, UNIT * 5])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')
       # infierno 4
        hell4_center = origin + np.array([UNIT * 8, UNIT * 5])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')
       # infierno 5
        hell5_center = origin + np.array([UNIT * 9, UNIT * 2])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')
       # infierno 6
        hell6_center = origin + np.array([UNIT * 10, UNIT * 10])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')
       # infierno 7
        hell7_center = origin + np.array([UNIT * 6, UNIT * 8])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')
       # infierno 8
        hell8_center = origin + np.array([UNIT * 5, UNIT * 8])
        self.hell8 = self.canvas.create_rectangle(
            hell8_center[0] - 15, hell8_center[1] - 15,
            hell8_center[0] + 15, hell8_center[1] + 15,
            fill='black')
       # infierno 9
        hell9_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.hell9 = self.canvas.create_rectangle(
            hell9_center[0] - 15, hell9_center[1] - 15,
            hell9_center[0] + 15, hell9_center[1] + 15,
            fill='black')
       # infierno 10
        hell10_center = origin + np.array([UNIT * 3, UNIT * 3])
        self.hell10 = self.canvas.create_rectangle(
            hell10_center[0] - 15, hell10_center[1] - 15,
            hell10_center[0] + 15, hell10_center[1] + 15,
            fill='black')

        # crear donde esta el oro
        oval_center1 = origin + UNIT * 4
        self.oval1 = self.canvas.create_oval(
            oval_center1[0] - 15, oval_center1[1] - 15,
            oval_center1[0] + 15, oval_center1[1] + 15,
            fill='yellow')

        oval_center2 = origin + np.array([UNIT * 7, UNIT * 7])
        self.oval2 = self.canvas.create_oval(
            oval_center2[0] - 15, oval_center2[1] - 15,
            oval_center2[0] + 15, oval_center2[1] + 15,
            fill='yellow')

        oval_center3 = origin + np.array([UNIT * 9, UNIT * 1])
        self.oval3 = self.canvas.create_oval(
            oval_center3[0] - 15, oval_center3[1] - 15,
            oval_center3[0] + 15, oval_center3[1] + 15,
            fill='yellow')

        # crear el rectangulo del agente
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='green')

        # tomar todos con método pack de canvas
        self.canvas.pack()

    def render(self):
        """
        Esperar para hacer update de la ventana
        """
        time.sleep(0.1)
        self.window.update()

    def reset(self):
        """
        Restablece el agente explorador en la posición de origen.
        Returns
        -------
            lienzo con el agente explorador en la posición de origen.
        """
        self.window.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='green')
        # return observation
        return self.canvas.coords(self.rect)

    def get_state_reward(self, action):
        """
        Restablece el agente explorador en la posición de origen.
        Parameters
        ----------
        action : TYPE
            DESCRIPTION.
        Returns
        -------
            Según donde esta parado retorna el estado siguiente
            recomepenza del agente, recompenza y si cagó el juego
        s_ : array
            donde esta parado.
        reward : int
            [-1, 0, 1].
        done : boolean
            sigue en el episodio o llega a estado terminal.
        """

        # obtener la coordenada del rectangulo, que es el agente
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        # en el caso que la acción sea 0, 1, 2, 3 las coordenadas se mueven
        if action == 0:
            # arriba
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:
            # abajo
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:
            # derecha
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            # izquierda
            if s[0] > UNIT:
                base_action[0] -= UNIT

        # mover el agente según la acción que ha tomado hacer
        self.canvas.move(
            self.rect, base_action[0], base_action[1])

        # actualizar el estado
        s_ = self.canvas.coords(self.rect)

        # función de recompenza
        # si está en las coordenadas del oro, dale uno
        if s_ in [self.canvas.coords(self.oval1),
                  self.canvas.coords(self.oval2),
                  self.canvas.coords(self.oval3)]:
            reward = 1
            done = True
            s_ = 'terminal'
        # si está en las coordenadas del infierno, dale menos uno
        elif s_ in [self.canvas.coords(self.hell1),
                    self.canvas.coords(self.hell2),
                    self.canvas.coords(self.hell3),
                    self.canvas.coords(self.hell4),
                    self.canvas.coords(self.hell5),
                    self.canvas.coords(self.hell6),
                    self.canvas.coords(self.hell7),
                    self.canvas.coords(self.hell8),
                    self.canvas.coords(self.hell9),
                    self.canvas.coords(self.hell10)]:
            reward = -1
            done = True
            # termina el juego
            s_ = 'terminal'
        else:
            # en todo lo demás dale tierra y la recompenza es cero
            reward = 0
            done = False
        return s_, reward, done
