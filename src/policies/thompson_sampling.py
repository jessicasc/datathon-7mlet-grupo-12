import numpy as np
import pandas as pd
from collections import defaultdict

from src.data.synthetic_generation import calculate_reward_probability

# politica contextual utilizando Thompson Sampling
class ThompsonSampling:

    def __init__(self, arms=(1, 2, 3, 4), random_state=None):

        self.arms = list(arms)

        self.alpha = defaultdict(
            lambda: {
                arm: 1
                for arm in self.arms
            }
        )

        self.beta = defaultdict(
            lambda: {
                arm: 1
                for arm in self.arms
            }
        )

        self.rng = np.random.default_rng(random_state)

    # segrega as features que serão usadas como contexto (ao todo serão 24 contextos)
    def get_context(self, customer):
        
        age_group = (
            "young"
            if customer["age"] < 25
            else "adult"
        )

        education_group = (
            "higher"
            if customer["education"] in [
                "university.degree",
                "professional.course"
            ]
            else "basic"
        )

        poutcome_group = (
            "success"
            if customer["poutcome"] == "success"
            else "no_success"
        )

        job_group = (
            "student"
            if customer["job"] in ["student"]
            else "management"
            if customer["job"] in ["management", "admin.", "entrepreneur"]
            else "other"
        )

        return (
            age_group,
            education_group,
            poutcome_group,
            job_group
        )

    def select_arm(self, context):

        samples = {
            arm: self.rng.beta(self.alpha[context][arm], self.beta[context][arm])
            for arm in self.arms
        }

        return max(samples, key=samples.get)

    def update(self, context, arm, reward):

        if reward == 1:
            self.alpha[context][arm] += 1
        else:
            self.beta[context][arm] += 1

    def run(self, customers: pd.DataFrame):

        results = []

        for _, customer in customers.iterrows():

            context = self.get_context(customer)

            chosen_arm = self.select_arm(context)

            p = calculate_reward_probability(customer, chosen_arm)

            reward = self.rng.binomial(1, p)

            self.update(context, chosen_arm, reward)

            results.append({
                "event_id": customer["event_id"],
                "customer_id": customer["customer_id"],
                "event_date": customer["event_date"],
                "chosen_arm": chosen_arm,
                "reward_probability": p,
                "reward": reward,
                "policy": "Contextual Thompson Sampling"
            })

        return pd.DataFrame(results)

    def recommend_arm(self, customer):

        context = self.get_context(customer)

        return self.select_arm(context)