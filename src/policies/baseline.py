import numpy as np
import pandas as pd

from src.data.synthetic_generation import calculate_reward_probability

# política deterministica que sempre seleciona o braço com maior taxa histórica de conversão
class DeterministicBaseline:

    def __init__(self, offer_events: pd.DataFrame):

        historical_conversion = (
            offer_events
            .groupby("arm_id")["reward"]
            .mean()
        )

        self.best_arm = historical_conversion.idxmax()

    def run(self, customers: pd.DataFrame) -> pd.DataFrame: 

        results = []

        for _, customer in customers.iterrows():

            p = calculate_reward_probability(customer, self.best_arm)

            reward = np.random.binomial(1, p)

            results.append({
                "event_id": customer["event_id"],
                "customer_id": customer["customer_id"],
                "event_date": customer["event_date"],
                "chosen_arm": self.best_arm,
                "reward_probability": p,
                "reward": reward,
                "policy": "Baseline"
            })

        return pd.DataFrame(results)