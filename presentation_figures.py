"""
presentation_figures.py draws the slide-ready HIKARI figures from the result files of integrate.py.
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
    hikari_detection()
    hikari_why_encrypted()
    hikari_duplicates()


def hikari_detection() -> None:
    table = pd.read_csv(HIKARI_RESULTS / "per_category.csv", index_col=0)
    names = {"Bruteforce": "Brute force (HTTPS)", "Bruteforce-XML": "Brute force XML (HTTPS)",
             "Probing": "Probing / scan (HTTPS)", "XMRIGCC CryptoMiner": "'Cryptominer' (NetBIOS chatter)"}
    table = table.loc[list(names)]
    figure, axis = plt.subplots(figsize=SIZE)
    _paired_bars(axis, [names[c] for c in table.index], table["suricata_flagged"], table["ml_flagged"])
    axis.set_title("HIKARI-2021: Suricata flags none of the labelled attacks", loc="left", fontsize=24, pad=34)
    axis.text(0, 1.04, f"Share of held-out attack flows flagged, per attack type ({int(table['flows'].sum()):,} flows, all encrypted or content-free)",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    _save(figure, "01_hikari_suricata_vs_model.png")


def hikari_why_encrypted() -> None:
    data = DataHandler.read_data()
    attacks = data[data["Label"] == 1]
    https = int((attacks["responp"] == 443).sum())
    netbios = int(attacks["responp"].isin([137, 138]).sum())
    figure, axis = plt.subplots(figsize=SIZE)
    axis.barh([0], [https], color=MODEL, height=0.5, label="HTTPS (TLS 1.3, port 443): content encrypted")
    axis.barh([0], [netbios], left=[https], color=NEITHER, height=0.5, label="NetBIOS broadcasts: no attack content")
    axis.text(https / 2, 0, f"{https:,} flows\n{https / len(attacks):.0%}", ha="center", va="center", color="white", fontsize=22, fontweight="bold")
    axis.text(https + netbios / 2, 0.38, f"{netbios:,}\n{netbios / len(attacks):.0%}", ha="center", va="bottom", fontsize=16)
    axis.set_ylim(-1.3, 1.3)
    axis.set_yticks([])
    axis.set_xlim(0, len(attacks))
    axis.set_xlabel("labelled attack flows in HIKARI-2021")
    axis.xaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    axis.grid(axis="y", visible=False)
    axis.set_title("Why Suricata sees nothing: no attack in HIKARI is readable", loc="left", fontsize=24, pad=34)
    axis.text(0, 1.04, "Every Bruteforce, Bruteforce-XML and Probing flow is TLS to port 443; ET Open's WordPress, WPScan and Nikto rules need plaintext HTTP",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False, fontsize=15)
    _save(figure, "02_hikari_why_zero.png")


def hikari_duplicates() -> None:
    data = DataHandler.read_data()
    key = ["originh", "originp", "responh", "responp", "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot"]
    benign = data[data["traffic_category"] == "Benign"]
    rows = []
    for category, label in [("Probing", "Probing"), ("Bruteforce-XML", "Brute force XML"), ("Bruteforce", "Brute force")]:
        attack = data[data["traffic_category"] == category]
        twins = attack.merge(benign[key].drop_duplicates(), on=key)
        rows.append((label, len(attack), len(twins[key].drop_duplicates())))
    figure, axis = plt.subplots(figsize=SIZE)
    labels = [name for name, _, _ in rows]
    positions = np.arange(len(rows))
    axis.bar(positions - 0.2, [total for _, total, _ in rows], width=0.38, color=NEITHER, label="attack rows in the CSV")
    axis.bar(positions + 0.2, [twin for _, _, twin in rows], width=0.38, color=SURICATA, label="of those, also listed as Benign (identical flow)")
    for position, (_, total, twin) in zip(positions, rows):
        axis.text(position - 0.2, total, f"{total:,}", ha="center", va="bottom", fontsize=15)
        axis.text(position + 0.2, twin, f"{twin:,}\n({twin / total:.0%})", ha="center", va="bottom", fontsize=15)
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, max(total for _, total, _ in rows) * 1.2)
    axis.yaxis.set_major_formatter(lambda value, _: f"{int(value):,}")
    axis.grid(axis="x", visible=False)
    axis.legend(frameon=False, loc="upper right", fontsize=15)
    axis.set_title("HIKARI-2021 lists most attack flows twice, once as Benign", loc="left", fontsize=24, pad=34)
    axis.text(0, 1.04, "Twins have identical features and endpoints (only the uid differs), so no detector can get both copies right",
              transform=axis.transAxes, color=INK_SECONDARY, fontsize=15)
    _save(figure, "03_hikari_duplicate_flows.png")


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
