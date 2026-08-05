import pandas as pd
from pathlib import Path
import random
import numpy as np
from datetime import datetime, timedelta
from src.data.loader import load_processed_dataset

START_DATE = datetime(2026, 1, 1)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[2]

OUTPUT = ROOT_DIR / "data" / "synthetic_enrichment"

# catalogo de braços com os tipos de abordagem para cada mensagem de oferta 
def create_offer_catalog():

    arms = pd.DataFrame(
        [
            {
                "arm_id": 1,
                "arm_name": "Formal",
                "message": "Conheça nosso novo CDB com rendimento de 110% do CDI e diversifique sua carteira."
            },
            {
                "arm_id": 2,
                "arm_name": "Educativa",
                "message": "Descubra como funciona um CDB e veja como ele pode ajudar você a investir com segurança."
            },
            {
                "arm_id": 3,
                "arm_name": "Descontraída",
                "message": "Seu dinheiro parado tá perdendo oportunidades 👀 Bora fazer ele render?"
            },
            {
                "arm_id": 4,
                "arm_name": "Benefícios",
                "message": "Faça seu dinheiro trabalhar para você, invista hoje no nosso novo CDB!"
            }
        ]
    )

    arms.to_csv(OUTPUT / "offer_catalog.sample.csv", index=False)

    print("Offer catalog criado com sucesso.")

# calcula a probabilidade do clique, baseada em hipoteses para cada braço
def calculate_reward_probability(customer, arm_id):

    p = 0.10

    # braço 1 - formal
    # ensino superior e cargos altos
    if arm_id == 1:

        if customer["job"] in ["management", "admin.", "entrepreneur"]:
            p += 0.15

        if customer["education"] in ["university.degree", "professional.course"]:
            p += 0.10

   # braço 2 - educativa
   # baixa escolaridade
    elif arm_id == 2:

        if customer["education"] in [
            "basic.4y",
            "basic.6y",
            "basic.9y",
        ]:
            p += 0.15

    # braço 3 - descontraída
    # jovens e estudantes
    elif arm_id == 3:

        if customer["age"] < 30:
            p += 0.20

        if customer["job"] == "student":
            p += 0.10

    # braço 4 - benefícios
    # histórico de sucesso em campanhas anteriores e maior número de contatos prévios
    elif arm_id == 4:

        if customer["poutcome"] == "success":
            p += 0.35

        if customer["previous"] >= 2:
            p += 0.10

    return min(p, 0.95)

# cria os eventos associando cliente, braço e recompensa
def create_offer_events():

    customers = load_processed_dataset()

    events = []

    event_id = 1

    for _, customer in customers.iterrows():

        # cada cliente recebe entre 1 e 5 mensagens
        n_events = np.random.randint(1, 6)

        for _ in range(n_events):

            arm = np.random.choice([1, 2, 3, 4])

            p = calculate_reward_probability(customer, arm)

            reward = np.random.binomial(1, p)

            # dia aleatório no mês de jan/26
            event_date = START_DATE + timedelta(days=np.random.randint(0, 31))

            events.append(
                {
                    "event_id": event_id,
                    "event_date": event_date.date(),
                    "customer_id": customer["customer_id"],
                    "age": customer["age"],
                    "job": customer["job"],
                    "education": customer["education"],
                    "poutcome": customer["poutcome"],
                    "previous": customer["previous"],
                    "arm_id": arm,
                    "reward": reward,
                }
            )

            event_id += 1

    events = pd.DataFrame(events)

    events.to_csv(OUTPUT / "offer_events.sample.csv", index=False)

    print("Offer events gerado.")

    return events

def generate_synthetic_data():

    print("Gerando catálogo de ofertas...")
    create_offer_catalog()

    print("Gerando eventos...")
    events = create_offer_events()

    print("Dados sintéticos gerados com sucesso!")

if __name__ == "__main__":
    generate_synthetic_data()