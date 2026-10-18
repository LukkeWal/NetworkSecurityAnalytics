"""
IntegrationReport compares Suricata with the ML model on the same flows. It reports
- the agreement table (both, Suricata only, ML only, neither)
- precision, recall and false positive rate per detector, plus how many Suricata false alerts the model removes
- detection per attack type
- the model's confusion matrix with and without background traffic
- Suricata against the model on all flows, on attacks only and on non-attacks only
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKGROUND_CATEGORY = "Background"
CELLS = {
    "both": (True, True),
    "suricata_only": (True, False),
    "ml_only": (False, True),
    "neither": (False, False),
}


def agreement_table(flows: pd.DataFrame) -> pd.DataFrame:
    """Count flows per agreement cell, split by what they truly are."""
    truth = _truth_group(flows)
    rows = {}
    for cell, (suricata, ml) in CELLS.items():
        in_cell = (flows["suricata_flag"] == suricata) & (flows["ml_flag"] == ml)
        rows[cell] = truth[in_cell].value_counts()
    table = pd.DataFrame(rows).T.fillna(0).astype(int)
    table = table.reindex(columns=["attack", "benign", BACKGROUND_CATEGORY.lower()], fill_value=0)
    table["total"] = table.sum(axis=1)
    return table


def detector_metrics(flows: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall and false positive rate for Suricata, the model and their combinations."""
    detectors = {
        "suricata": flows["suricata_flag"],
        "ml": flows["ml_flag"],
        "suricata_and_ml": flows["suricata_flag"] & flows["ml_flag"],
        "suricata_or_ml": flows["suricata_flag"] | flows["ml_flag"],
    }
    attack = flows["Label"] == 1
    rows = {}
    for name, flagged in detectors.items():
        tp, fp = int((flagged & attack).sum()), int((flagged & ~attack).sum())
        fn, tn = int((~flagged & attack).sum()), int((~flagged & ~attack).sum())
        rows[name] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "false_positive_rate": _ratio(fp, fp + tn),
        }
    return pd.DataFrame(rows).T


def false_alerts_removed(flows: pd.DataFrame) -> dict[str, float]:
    """What happens to Suricata's alerts when only those the model confirms are kept."""
    attack = flows["Label"] == 1
    alerts = flows["suricata_flag"]
    false_alerts = alerts & ~attack
    true_alerts = alerts & attack
    return {
        "suricata_false_alerts": int(false_alerts.sum()),
        "false_alerts_removed": int((false_alerts & ~flows["ml_flag"]).sum()),
        "suricata_true_alerts": int(true_alerts.sum()),
        "true_alerts_lost": int((true_alerts & ~flows["ml_flag"]).sum()),
    }


def per_category(flows: pd.DataFrame) -> pd.DataFrame:
    """Share of flows each detector flags, per traffic category."""
    grouped = flows.groupby("traffic_category")
    return pd.DataFrame({
        "flows": grouped.size(),
        "suricata_flagged": grouped["suricata_flag"].mean(),
        "ml_flagged": grouped["ml_flag"].mean(),
        "both_flagged": grouped.apply(lambda g: (g["suricata_flag"] & g["ml_flag"]).mean()),
    })


def ml_confusion_matrices(flows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rows are the true class (benign, attack), columns the model's decision (benign, attack)."""
    without_background = flows[flows["traffic_category"] != BACKGROUND_CATEGORY]
    return {
        "all flows": _confusion(flows),
        "without background": _confusion(without_background),
    }


def suricata_vs_ml_matrices(flows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Rows are Suricata's decision, columns the model's (not flagged, flagged)."""
    attack = flows["Label"] == 1
    return {
        "all flows": _agreement(flows),
        "attacks": _agreement(flows[attack]),
        "benign + background": _agreement(flows[~attack]),
    }


def save_report(flows: pd.DataFrame, out_dir: Path | str) -> None:
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    agreement = agreement_table(flows)
    metrics = detector_metrics(flows)
    categories = per_category(flows)
    agreement.to_csv(out_dir / "agreement.csv")
    metrics.to_csv(out_dir / "detector_metrics.csv")
    categories.to_csv(out_dir / "per_category.csv")
    pd.Series(false_alerts_removed(flows)).to_csv(out_dir / "false_alerts_removed.csv", header=False)
    flows.to_csv(out_dir / "flows_scored.csv", index=False)

    for name, matrices in {"suricata_vs_ml": suricata_vs_ml_matrices(flows)}.items():
        pd.concat({title: pd.DataFrame(matrix, index=["suricata_no", "suricata_yes"], columns=["ml_no", "ml_yes"])
                   for title, matrix in matrices.items()}).to_csv(out_dir / f"{name}_matrices.csv")

    plot_agreement(agreement).savefig(figures_dir / "agreement.png", dpi=150, bbox_inches="tight")
    plot_confusion_matrices(ml_confusion_matrices(flows)).savefig(
        figures_dir / "ml_confusion_matrices.png", dpi=150, bbox_inches="tight"
    )
    plot_confusion_matrices(
        suricata_vs_ml_matrices(flows), ["not flagged", "flagged"], "model", "Suricata"
    ).savefig(figures_dir / "suricata_vs_ml_matrices.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def plot_agreement(table: pd.DataFrame):
    figure, axis = plt.subplots(figsize=(8, 5))
    table.drop(columns="total").plot(kind="bar", stacked=True, ax=axis, rot=0)
    axis.set_yscale("symlog")
    axis.set_ylabel("flows")
    axis.set_title("Suricata vs ML model")
    return figure


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray],
    labels: list[str] | None = None,
    xlabel: str = "predicted",
    ylabel: str = "true",
):
    figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5))
    for axis, (title, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], labels or ["benign", "attack"])
        axis.set_yticks([0, 1], labels or ["benign", "attack"])
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        for (row, column), count in np.ndenumerate(matrix):
            axis.text(column, row, f"{count:,}", ha="center", va="center",
                      color="white" if count > matrix.max() / 2 else "black")
    figure.tight_layout()
    return figure


def _truth_group(flows: pd.DataFrame) -> pd.Series:
    group = pd.Series("benign", index=flows.index)
    group[flows["traffic_category"] == BACKGROUND_CATEGORY] = BACKGROUND_CATEGORY.lower()
    group[flows["Label"] == 1] = "attack"
    return group


def _confusion(flows: pd.DataFrame) -> np.ndarray:
    attack = (flows["Label"] == 1).to_numpy()
    flagged = flows["ml_flag"].to_numpy()
    return np.array([
        [(~attack & ~flagged).sum(), (~attack & flagged).sum()],
        [(attack & ~flagged).sum(), (attack & flagged).sum()],
    ])


def _agreement(flows: pd.DataFrame) -> np.ndarray:
    suricata = flows["suricata_flag"].to_numpy()
    ml = flows["ml_flag"].to_numpy()
    return np.array([
        [(~suricata & ~ml).sum(), (~suricata & ml).sum()],
        [(suricata & ~ml).sum(), (suricata & ml).sum()],
    ])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")
