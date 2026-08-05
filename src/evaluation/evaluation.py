import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path

from src.policies.baseline import DeterministicBaseline
from src.policies.thompson_sampling import ThompsonSampling

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# metricas por politica
def calculate_metrics(results: pd.DataFrame) -> dict:

    total_events = len(results)
    total_reward = results["reward"].sum()
    conversion_rate = total_reward / total_events

    exploration_rate = (results["chosen_arm"].nunique() / 4)

    return {
        "total_events": total_events,
        "total_reward": total_reward,
        "conversion_rate": conversion_rate,
        "exploration_rate": exploration_rate
    }

# compara as metricas entre as politicas
def compare_policies(baseline_results: pd.DataFrame, thompson_results: pd.DataFrame) -> pd.DataFrame:

    baseline = calculate_metrics(baseline_results)
    thompson = calculate_metrics(thompson_results)

    comparison = pd.DataFrame({
        "Baseline": [
            baseline["total_events"],
            baseline["total_reward"],
            baseline["conversion_rate"] * 100,
            baseline["exploration_rate"] * 100
        ],
        "Thompson": [
            thompson["total_events"],
            thompson["total_reward"],
            thompson["conversion_rate"] * 100,
            thompson["exploration_rate"] * 100
        ]
    },
    index=[
        "Total events",
        "Reward",
        "Conversion Rate (%)",
        "Exploration Rate (%)"
    ])

    return comparison.round(2)

# compara reward total por politica
def plot_reward_comparison(baseline_results: pd.DataFrame,thompson_results: pd.DataFrame):

    rewards = {
        "Baseline": baseline_results["reward"].sum(),
        "Thompson": thompson_results["reward"].sum()
    }

    plt.figure(figsize=(6,4))
    plt.bar(rewards.keys(), rewards.values())
    plt.title("Reward total por política")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "reward_comparison.png")
    plt.close()

# compara a distribuição dos braços escolhidos
def plot_arm_distribution(baseline_results: pd.DataFrame,thompson_results: pd.DataFrame):

    baseline = (
        baseline_results["chosen_arm"]
        .value_counts()
        .sort_index()
    )

    thompson = (
        thompson_results["chosen_arm"]
        .value_counts()
        .sort_index()
    )

    comparison = pd.DataFrame({
        "Baseline": baseline,
        "Thompson": thompson
    }).fillna(0)

    comparison.plot(
        kind="bar",
        figsize=(8,4)
    )

    plt.title("Distribuição de escolhas por braço")
    plt.xlabel("Braço")
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "arm_distribution.png")
    plt.close()

# compara a taxa de conversão entre as políticas
def plot_conversion_rate(baseline_results: pd.DataFrame,thompson_results: pd.DataFrame):

    conversion = {
        "Baseline": (
            baseline_results["reward"].sum()
            / len(baseline_results)
        ),
        "Thompson": (
            thompson_results["reward"].sum()
            / len(thompson_results)
        )
    }

    plt.figure(figsize=(6,4))
    plt.bar(conversion.keys(), conversion.values())
    plt.title("Taxa de Conversão")
    plt.ylabel("Conversion Rate")

    # mostra o valor em porcentagem em cima das barras
    for i, value in enumerate(conversion.values()):
        plt.text(
            i,
            value,
            f"{value:.2%}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "conversion_rate_comparison.png")
    plt.close()

# executa toda validação das politicas
def run_evaluation():

    offer_events = pd.read_csv(
        "data/synthetic_enrichment/offer_events.sample.csv",
        parse_dates=["event_date"]
    )

    print("Executando política baseline...")
    baseline = DeterministicBaseline(offer_events)
    baseline_results = baseline.run(offer_events)

    print("Executando política Thompson Sampling...")
    thompson = ThompsonSampling()
    thompson_results = thompson.run(offer_events)

    print("Comparação entre as políticas:\n")

    comparison = compare_policies(baseline_results,thompson_results)

    print(comparison)

    print("\nMelhor política: ", comparison.loc["Reward"].idxmax())

    print("\nGerando gráficos...")

    plot_reward_comparison(baseline_results,thompson_results)

    plot_arm_distribution(baseline_results,thompson_results)

    plot_conversion_rate(baseline_results,thompson_results)

    print("\nGráficos salvos em:", OUTPUT_DIR)

    # MLFlow
    thompson_metrics = calculate_metrics(thompson_results)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Datathon Offer Recommendation")

    with mlflow.start_run():

        mlflow.log_param("policy", "Contextual Thompson Sampling")

        mlflow.log_param("arms", 4)

        mlflow.log_param("context_features","age_group, education_group, job_group, poutcome_group")

        mlflow.log_param("contexts",24)

        mlflow.log_param("random_state", 42)

        mlflow.log_metric("total_events", thompson_metrics["total_events"])

        mlflow.log_metric("reward", thompson_metrics["total_reward"])

        mlflow.log_metric("conversion_rate", thompson_metrics["conversion_rate"])

        mlflow.log_metric("exploration_rate", thompson_metrics["exploration_rate"])

        mlflow.log_artifact(OUTPUT_DIR / "reward_comparison.png")

        mlflow.log_artifact(OUTPUT_DIR / "arm_distribution.png")

        mlflow.log_artifact(OUTPUT_DIR / "conversion_rate_comparison.png")

if __name__ == "__main__":
    run_evaluation()