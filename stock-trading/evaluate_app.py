import matplotlib.pyplot as plt
from keras.models import load_model
from src.agent import Agent
from src.market_env import Market


def main():
    """
    Evaluar el agente entrenado en un dataset de acciones en otro
    completamente diferente
    """
    stock_name = "GSPC_2011-03"
    model_name = "model_rl"
    # cargar pesos
    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]

    agent = Agent(window_size, True, model_name)
    market = Market(window_size, stock_name)

    # Empezar desde un estado inicial
    state, price_data = market.reset()

    for t in range(market.last_data_index):

        # accion para el estado actual
        action, bought_price = agent.act(state, price_data)
        # verificar la acción para obtener recompensa y observar
        # el siguiente estado
        # obtener el siguiente estado
        next_state, next_price_data, reward, done =\
            market.get_next_state_reward(action, bought_price)
        # estado siguiente y ganancias totales
        state = next_state
        price_data = next_price_data
        if done:
            print("--------------------------------")
            print("{0} Ganancias totales: {1}".format(
                stock_name, agent.get_total_profit()))
            print("--------------------------------")
    plot_action_profit(market.data, agent.action_history,
                       agent.get_total_profit())


def plot_action_profit(data, action_data, profit):
    """
    Plot de las acciones realizadas
    Parameters
    ----------
    data : list
        DESCRIPTION.
    action_data : int
        DESCRIPTION.
    profit : float
        DESCRIPTION.
    """
    plt.plot(range(len(data)), data)
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    buy, sel = False, False
    for d in range(len(data) - 1):
        # comprar
        if action_data[d] == 1:
            buy, = plt.plot(d, data[d], 'g*')
        # vender
        elif action_data[d] == 2:  # sell
            sel, = plt.plot(d, data[d], 'r+')
    if buy and sel:
        plt.legend([buy, sel], ["Compras", "Ventas"])
    plt.title("Ganancias totaoles: {0}".format(profit))
    plt.savefig("ventas_compras.png")
    plt.show()


if __name__ == "__main__":
    main()
