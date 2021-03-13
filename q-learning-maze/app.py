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
    Reinforcement learning maze example.
    This function updates the position of the explorer in the Maze
    environment based on the actions it chooses.

    This script is the main part which controls the update method of
    this example using q-learning algorithm.
    The RL algorithm (Q-learning) is in RL_agent.py.
    The environment is presented in maze_env.py.
    Red rectangle:          explorer.
    Black rectangles:       hells       [reward = -1].
    Yellow bin circle:      paradise    [reward = +1].
    All other states:       ground      [reward = 0].
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
            # romper mientras bucle al final de este episodio
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
    Resultados del aprendizaje por Q-Table
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
