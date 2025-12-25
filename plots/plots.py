"""
Plot success rates for introspection experiments.

Success definitions:
    - anthropic_reproduce:
        coherence AND affirmative_response_followed_by_correct_identification

    - anthropic_reproduce_binary:
        coherence AND binary_detection

    - mcq_knowledge:
        coherence AND mcq_correct

    - mcq_distinguish:
        coherence AND mcq_correct

    - open_ended_belief:
        coherence AND thinking_about_word

    - generative_distinguish:
        coherence ONLY (no correctness judge exists yet)

    - injection_strength:
        coherence AND injection_strength_correct
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def compute_success_rate(df, experiment_type):
    """
    Compute success rate per (layer, coeff, vec_type) combination.
    """
    success_rates = defaultdict(lambda: defaultdict(dict))

    for layer in sorted(df["layer"].unique()):
        for coeff in sorted(df["coeff"].unique()):
            for vec_type in sorted(df["vec_type"].unique()):
                subset = df[
                    (df["layer"] == layer) &
                    (df["coeff"] == coeff) &
                    (df["vec_type"] == vec_type)
                ]

                if subset.empty:
                    continue

                # ---- SUCCESS LOGIC ----
                if experiment_type == "anthropic_reproduce":
                    successes = (
                        subset["coherence_judge"] &
                        subset["affirmative_response_followed_by_correct_identification_judge"]
                    )

                elif experiment_type == "anthropic_reproduce_binary":
                    successes = (
                        subset["coherence_judge"] &
                        subset["binary_detection_judge"]
                    )

                elif experiment_type in ["mcq_knowledge", "mcq_distinguish"]:
                    successes = (
                        subset["coherence_judge"] &
                        subset["mcq_correct_judge"]
                    )

                elif experiment_type == "open_ended_belief":
                    successes = (
                        subset["coherence_judge"] &
                        subset["thinking_about_word_judge"]
                    )

                elif experiment_type == "generative_distinguish":
                    # No correctness judge exists — coherence only
                    successes = subset["coherence_judge"]

                elif experiment_type == "injection_strength":
                    successes = (
                        subset["coherence_judge"] &
                        subset["injection_strength_correct_judge"]
                    )

                else:
                    raise ValueError(f"Unknown experiment type: {experiment_type}")

                # ---- SAFE AGGREGATION ----
                successes = successes.fillna(False)
                rate = successes.mean()

                success_rates[layer][coeff][vec_type] = rate

    return success_rates


def plot_success_rates(success_rates, experiment_type, output_dir):
    """
    Plot success rates with layer on x-axis and separate lines for each (coeff, vec_type).
    """
    layers = sorted(success_rates.keys())

    coeff_vec_pairs = sorted({
        (coeff, vec_type)
        for layer_data in success_rates.values()
        for coeff, vec_dict in layer_data.items()
        for vec_type in vec_dict.keys()
    })

    plt.figure(figsize=(12, 7))

    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10(range(len(coeff_vec_pairs)))

    for i, (coeff, vec_type) in enumerate(coeff_vec_pairs):
        rates = [
            success_rates[layer].get(coeff, {}).get(vec_type, 0.0)
            for layer in layers
        ]

        plt.plot(
            layers,
            rates,
            marker=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i],
            label=f"Coeff {coeff}, {vec_type}",
            linewidth=2,
            markersize=6,
        )

    ylabel_map = {
        "anthropic_reproduce":
            "Success Rate (Coherence ∧ Affirmative + Correct Identification)",
        "anthropic_reproduce_binary":
            "Success Rate (Coherence ∧ Binary Detection)",
        "mcq_knowledge":
            "Success Rate (Coherence ∧ MCQ Correct)",
        "mcq_distinguish":
            "Success Rate (Coherence ∧ MCQ Correct)",
        "open_ended_belief":
            "Success Rate (Coherence ∧ Thinking About Word)",
        "generative_distinguish":
            "Success Rate (Coherence Only)",
        "injection_strength":
            "Success Rate (Coherence ∧ Injection Strength Correct)",
    }

    title_map = {
        "anthropic_reproduce":
            "Do you detect an injected thought? If so, what is it about?",
        "anthropic_reproduce_binary":
            "Do you detect an injected thought? (Binary Response Only)",
        "mcq_knowledge":
            "Which concept is influencing your thoughts? (10-way MCQ)",
        "mcq_distinguish":
            "Which concept is influencing your thoughts? (2-way MCQ)",
        "open_ended_belief":
            "What concept are you thinking about right now?",
        "generative_distinguish":
            "Which phenomenon is true?",
        "injection_strength":
            "What is the strength of the injected thought?",
    }

    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(ylabel_map.get(experiment_type, "Success Rate"), fontsize=12)
    plt.title(title_map.get(experiment_type, experiment_type), fontsize=14)
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.8)
    plt.tight_layout()

    output_path = output_dir / f"success_rate_{experiment_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    """
    Generate plots for all experiment types.
    """
    results_dir = Path(__file__).parent.parent / "new_results"
    output_dir = Path(__file__).parent

    experiment_types = [
        "anthropic_reproduce",
        "anthropic_reproduce_binary",
        "mcq_knowledge",
        "mcq_distinguish",
        "open_ended_belief",
        "generative_distinguish",
        "injection_strength",
    ]

    for exp_type in experiment_types:
        for scope in ["assistant", "all_tokens"]:
            csv_path = results_dir / f'output_{exp_type}_{scope}.csv'

            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping...")
                continue

            print(f"\nProcessing {exp_type} ({scope})...")
            df = pd.read_csv(csv_path)

            success_rates = compute_success_rate(df, exp_type)
            plot_success_rates(success_rates, f"{exp_type}_{scope}", output_dir)

        print(f"\nProcessing {exp_type}...")
        df = pd.read_csv(csv_path)

        # ---- SAFE BOOLEAN COERCION ----
        bool_cols = [
            "coherence_judge",
            "thinking_about_word_judge",
            "affirmative_response_judge",
            "affirmative_response_followed_by_correct_identification_judge",
            "binary_detection_judge",
            "mcq_correct_judge",
            "injection_strength_correct_judge",
        ]

        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: True if x is True or x == "True" else False
                )

        success_rates = compute_success_rate(df, exp_type)
        plot_success_rates(success_rates, exp_type, output_dir)


if __name__ == "__main__":
    main()
