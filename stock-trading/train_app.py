import os
import time
import warnings
from src.agent import Agent
from src.market_env import Market
warnings.filterwarnings("ignore")


def main():

    window_size = 5
    episode_count = 10
    stock_name = "^GSPC_2011"

    agent = Agent(window_size)
    market = Market(window_size=window_size, stock_name=stock_name)

    batch_size = 32

    start_time = time.time()
    for e in range(episode_count + 1):
        print("Episodio" + str(e) + "/" + str(episode_count))
        agent.reset()
        state, price_data = market.reset()  # ToDo: get the initial state

        for t in range(market.last_data_index):
            # obtener acción actual del agente
            # llamar al método act() del agente considerando el estado actual
            action, bought_price = agent.act(state, price_data)

            # obtener siguiente estado del agente según el mercado
            next_state, next_price_data, reward, done =\
                market.get_next_state_reward(action, bought_price)

            # añadir trasacción a la memoria
            agent.memory.append((state, action, reward, next_state, done))
            # aprender de la historia solo en el caso que haya memoria
            if len(agent.memory) > batch_size:
                agent.experience_replay(batch_size)

            state = next_state
            price_data = next_price_data

            if done:
                print("--------------------------------")
                print("Ganancias totales: {0}".format(
                    agent.get_total_profit()))
                print("--------------------------------")

        if e % 10 == 0:
            if not os.path.exists("models"):
                os.mkdir("models")
            agent.model.save("models/model_rl" + str(e))

    end_time = time.time()
    training_time = round(end_time - start_time)
    print("Entrenamiento tomó {0} segundos.".format(training_time))


if __name__ == "__main__":
    main()
