import matplotlib
import matplotlib.pyplot as plt
from src.maze_env import MazeEnviroment
from src.rl_agent import QLearningTable
matplotlib.use('TkAgg')

# Parámetros
# número de episodios que van hacer
episode_count = 100
episodes = range(episode_count)


# FIX: agregar un filtro a la lista de los últimos 1000 valores, para no
# comerme la ram si agrando el enviroment

# Número de movimientos pasados en cada episodio
movements = []
# las recompenzas ganadas en cada episodio
rewards = []


def run_experiment():
    """
    Salir del laberinto con aprendizaje por refuerzo.
    Esta función actualiza la posición del explorador en el Laberinto.
    entorno basado en las acciones que elige.
    Este script es la parte principal que controla el método de
    actualización usando el algoritmo q-learning.

    El algoritmo RL (Q-learning) esta en src/rl_agent.py.
    El entorno de esta en src/maze_env.py.
    Rectangulo verde:        agente.
    rectangulo negro:       perdiste/te caiste     [reward = -1].
    circulo amarillo:      encontro oro           [reward = +1].
    Todos los demás:       tierra                 [reward = 0].
    """
    for episode in episodes:
        print("Episodio: ", f"{episode +1}/{episode_count}")

        # observación inicial
        observation = env.reset()
        moves = 0
        while True:
            # tabla limpia
            env.render()
            # Q-learning escoge acciones basadas en observaciones
            # convertimos la observación a str ya que queremos usarlos como
            # índice para nuestro DataFrame.
            # llamar al método choose_action () desde el agente QLearningTable
            action = q_learning_agent.choose_action(str(observation))
            # RL hace acciones y tiene el siguiente estado y recompenza
            observation_, reward, done = env.get_state_reward(action)
            moves += 1
            # agente aprende del estado anterior
            # Actualizar Q value para la tupla de valores actual
            # (s, a, r, s') tupla
            # llamar al método de aprendizaje de la instancia del agente de
            # Q-learning, pasando
            q_learning_agent.learn(str(observation), action, reward,
                                   str(observation_))
            observation = observation_
            # romper cuando finalice episodio (1 o -1)
            if done:
                # guardar el número de movimientos hechos para ganar
                movements.append(moves)
                # gardar las recompenzas
                rewards.append(reward)
                print(
                    "Recompenza: {0}, Movimientos realizados: {1}".format(
                        reward, moves))
                break
    # terminal juego
    print('Jueegueee !')
    # Mostrar los resultados del aprendizaje
    plot_reward_movements()


def plot_reward_movements():
    """
    Resultados del aprendizaje por Q-reinforcement learning
    Returns
    -------
    Plot de como aprendió
    """
    titulo = "Resultado de aprendizaje RL Agent: Movimientos"
    plt.title(titulo, fontsize=30)
    fig, ax = plt.subplots(1, figsize=(22, 12))
    plt.scatter(episodes, movements, color='blue', zorder=1)
    plt.xlabel('Episodios', fontsize=30)
    plt.ylabel('Movimientos', fontsize=30)
    plt.legend(
        ["Movimientos realizados"], fontsize=22, loc="upper left")
    plt.show()
    fig.savefig("images/rl_movimientos.png")

    titulo = "Resultado de aprendizaje RL Agent: Recompenzas"
    plt.title(titulo, fontsize=30)
    fig, ax = plt.subplots(1, figsize=(22, 12))
    plt.scatter(episodes, rewards, color='green', zorder=1)
    plt.xlabel('Episodios', fontsize=30)
    plt.ylabel('Recompenzas', fontsize=30)
    plt.legend(
        ["Recompenza obtenida"], fontsize=22, loc="upper left")
    plt.show()
    fig.savefig("images/rl_recompenzas.png")


if __name__ == "__main__":

    # crear el ambiente
    env = MazeEnviroment()
    # crear Q-learning agent
    q_learning_agent = QLearningTable(actions=list(range(env.n_actions)))

    # llamar la función run_experiment()
    env.window.after(10, run_experiment)
    env.window.mainloop()
