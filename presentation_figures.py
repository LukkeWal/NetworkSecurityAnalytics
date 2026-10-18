"""
presentation_figures.py draws the slide-ready CSE-CIC-IDS2018 figures from the result files of integrate_cic.py.
The cross-dataset figure also needs the HIKARI model (python integrate.py train).
Run it from the project directory; the images go to presentation_figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DataHandler
import ModelStore

HIKARI_RESULTS = Path("integration_results")
CIC_RESULTS = Path("integration_results_cic")
OUTPUT_PATH = Path("presentation_figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
MODEL = "#2a78d6"
SURICATA = "#eb6834"
BOTH = "#1baf7a"
NEITHER = "#b9b8b2"
SIZE = (13.33, 7.5)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_SECONDARY, "xtick.color": INK_SECONDARY, "ytick.color": INK,
    "font.size": 16, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": GRID, "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
})


def main() -> None:
    OUTPUT_PATH.mkdir(exist_ok=True)
    cic_detection()
    cic_overlap()
    cic_cross_dataset()
    cic_unseen()
    cic_sweep()


def cic_detection() -> None:
    table = pd.read_csv(CIC_RESULTS / "per_category.csv", index_col=0)
    attacks = table.drop(index="Benign").sort_values("suricata_flagged", ascending=True)
    figure, axis = plt.subplots(figsize=SIZE)
    labels = [f"{name} (n={int(flows):,})" for name, flows in zip(attacks.index, attacks["flows"])]
    _paired_bars(axis, labels, attacks["suricata_flagged"], attacks["ml_flagged"])
    axis.set_title("CSE-CIC-IDS2018: each tool catches what the other cannot", loc="left", fontsize=24, pad=34)
    axis.text(0, 1.04, "Share of held-out test flows flagged, per attack type. Suricata wins on web attacks, the model on floods and brute force",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    _save(figure, "04_cic_suricata_vs_model.png")


def cic_overlap() -> None:
    matrices = pd.read_csv(CIC_RESULTS / "suricata_vs_ml_matrices.csv", index_col=[0, 1])
    attacks = matrices.loc["attacks"]
    normal = matrices.loc["benign + background"]
    parts = [
        ("Both flag it", attacks.loc["suricata_yes", "ml_yes"], BOTH),
        ("Only the model", attacks.loc["suricata_no", "ml_yes"], MODEL),
        ("Only Suricata", attacks.loc["suricata_yes", "ml_no"], SURICATA),
        ("Neither", attacks.loc["suricata_no", "ml_no"], NEITHER),
    ]
    total = sum(count for _, count, _ in parts)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=SIZE, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.9})
    left = 0
    for name, count, color in parts:
        top.barh([0], [count], left=[left], color=color, height=0.55)
        if count / total > 0.08:
            top.text(left + count / 2, 0, f"{count:,}", ha="center", va="center", color="white" if color != NEITHER else INK, fontsize=20, fontweight="bold")
        left += count
    top.set_xlim(0, total)
    top.set_yticks([])
    top.grid(visible=False)
    top.set_title(f"Of {total:,} real attacks in the test set:", loc="left", fontsize=18, color=INK_SECONDARY)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, _, color in parts]
    top.legend(handles, [f"{name}: {count:,}" for name, count, _ in parts], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=15)
    false_alarms = [("Model false alarms", normal.loc["suricata_no", "ml_yes"], MODEL), ("Suricata false alarms", normal.loc["suricata_yes", "ml_no"], SURICATA),
                    ("Both", normal.loc["suricata_yes", "ml_yes"], BOTH)]
    normal_total = int(normal.to_numpy().sum())
    labels = [f"{name}: {int(count):,}" for name, count, _ in false_alarms]
    bottom.barh(range(len(false_alarms)), [count for _, count, _ in false_alarms], color=[color for _, _, color in false_alarms], height=0.55)
    bottom.set_yticks(range(len(false_alarms)), labels)
    bottom.invert_yaxis()
    bottom.set_xlim(0, max(1, max(count for _, count, _ in false_alarms) * 1.15))
    bottom.xaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    bottom.grid(axis="y", visible=False)
    bottom.set_title(f"Of {normal_total:,} normal flows, wrongly flagged by:", loc="left", fontsize=18, color=INK_SECONDARY)
    figure.suptitle("CSE-CIC-IDS2018: the tools overlap and complement each other", x=0.06, ha="left", fontsize=24, y=0.99)
    _save(figure, "05_cic_overlap.png")


def cic_unseen() -> None:
    table = pd.read_csv(CIC_RESULTS / "unseen_attack_types.csv").sort_values("model_flags_it")
    figure, axis = plt.subplots(figsize=SIZE)
    _paired_bars(axis, [f"{name} (n={int(flows):,})" for name, flows in zip(table["unseen_attack"], table["flows"])],
                 table["suricata_flags_it"], table["model_flags_it"])
    axis.set_title("A model that never saw the attack mostly misses it; Suricata does not depend on training", loc="left", fontsize=22, pad=34)
    axis.text(0, 1.04, "Model retrained without each attack type in turn, then asked to flag it. Suricata's rules are unaffected",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    _save(figure, "06_cic_unseen_attacks.png")


def cic_cross_dataset() -> None:
    flows = pd.read_pickle(CIC_RESULTS / "flows_labelled.pkl")
    attacks = flows[flows["Label"] == 1]
    hikari_model = ModelStore.load()
    transferred = (ModelStore.score(hikari_model, attacks) >= hikari_model.threshold).mean()
    detector = pd.read_csv(CIC_RESULTS / "detector_metrics.csv", index_col=0)
    bars = [("Suricata + ET Open\n(no training needed)", detector.loc["suricata", "recall"], SURICATA),
            ("Model trained on HIKARI,\nused on 2018 attacks", transferred, NEITHER),
            ("Model trained on 2018,\nused on 2018 test flows", detector.loc["ml", "recall"], MODEL)]
    figure, axis = plt.subplots(figsize=SIZE)
    axis.bar(range(3), [value for _, value, _ in bars], color=[color for _, _, color in bars], width=0.55)
    for position, (_, value, _) in enumerate(bars):
        axis.text(position, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=24, fontweight="bold")
    axis.set_xticks(range(3), [name for name, _, _ in bars])
    axis.set_ylim(0, 1.12)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.grid(axis="x", visible=False)
    axis.set_ylabel("share of attack flows flagged")
    axis.set_title("The model does not transfer between datasets", loc="left", fontsize=24, pad=34)
    axis.text(0, 1.04, "HIKARI-trained model on CSE-CIC-IDS2018 attacks: flags nothing. It must be retrained for each network",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    _save(figure, "07_cross_dataset.png")


def cic_sweep() -> None:
    table = pd.read_csv(CIC_RESULTS / "sweep_results.csv")
    table = table[table["feature_representation_name"] == "raw_numeric"].drop_duplicates("classifier_name", keep="last")
    table = table.sort_values("f1")
    figure, axis = plt.subplots(figsize=SIZE)
    positions = np.arange(len(table))
    axis.barh(positions + 0.2, table["recall"], height=0.38, color=MODEL, label="recall (attacks caught)")
    axis.barh(positions - 0.2, table["precision"], height=0.38, color=BOTH, label="precision (alerts that are real)")
    for position, recall, precision, fpr in zip(positions, table["recall"], table["precision"], table["false_positive_rate"]):
        axis.text(recall + 0.01, position + 0.2, f"{recall:.0%}", va="center", fontsize=14)
        axis.text(precision + 0.01, position - 0.2, f"{precision:.0%}   (false alarm rate {fpr:.1%})", va="center", fontsize=14)
    axis.set_yticks(positions, table["classifier_name"])
    axis.set_xlim(0, 1.3)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}" if value <= 1 else "")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=15)
    axis.set_title("Classifier comparison on CSE-CIC-IDS2018 (threshold set for 95% recall)", loc="left", fontsize=22, pad=34)
    axis.text(0, 1.04, "Same method as the project's main.py sweep; ordered by F1",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    _save(figure, "08_cic_classifier_sweep.png")


def _percent(value: float) -> str:
    """Whole percent, except just under 100% where rounding would hide the difference."""
    return f"{value:.1%}" if 0.985 <= value < 1 else f"{value:.0%}"


def _paired_bars(axis, labels, suricata, model) -> None:
    positions = np.arange(len(labels))
    axis.barh(positions + 0.2, model, height=0.38, color=MODEL, label="ML model")
    axis.barh(positions - 0.2, suricata, height=0.38, color=SURICATA, label="Suricata + ET Open")
    for position, share_model, share_suricata in zip(positions, model, suricata):
        axis.text(share_model + 0.01, position + 0.2, _percent(share_model), va="center", fontsize=15, fontweight="bold")
        axis.text(share_suricata + 0.01, position - 0.2, _percent(share_suricata), va="center", fontsize=15, fontweight="bold")
    axis.set_yticks(positions, labels)
    axis.set_xlim(0, 1.12)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}" if value <= 1 else "")
    axis.grid(axis="y", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=16)


def _save(figure, name: str) -> None:
    figure.savefig(OUTPUT_PATH / name, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print("saved", OUTPUT_PATH / name)


if __name__ == "__main__":
    main()
