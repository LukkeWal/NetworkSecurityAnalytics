"""
Builds demo/results.html from the saved results (CSVs and figures already in the repository).

    python demo/build_report.py
    python demo/build_report.py --open
"""

import base64
import html
import sys
import webbrowser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results.html"
LAB = ROOT / "tls-inspection-lab" / "runs" / "full" / "results"


def image(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def table(frame: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in frame.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def figure(name: str, caption: str) -> str:
    return f'<figure><img src="{image(ROOT / "presentation_figures" / name)}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def card(number: str, label: str, detail: str) -> str:
    return f'<div class="card"><div class="num">{number}</div><div class="lab">{label}</div><div class="det">{detail}</div></div>'


def build() -> str:
    hikari = pd.read_csv(ROOT / "integration_results" / "detector_metrics.csv", index_col=0)
    cic = pd.read_csv(ROOT / "integration_results_cic" / "detector_metrics.csv", index_col=0)
    removed = pd.read_csv(ROOT / "integration_results_cic" / "false_alerts_removed.csv", header=None, index_col=0)[1]
    lab = pd.read_csv(LAB / "summary.csv")
    attacks = lab[lab["category"] != "Benign"]
    n_attack = int(attacks["flows"].sum())
    lab_dec = (attacks["flows"] * attacks["suricata_decrypted"]).sum() / n_attack
    hikari_attacks = int(hikari.loc["ml", "tp"] + hikari.loc["ml", "fn"])
    cic_attacks = int(cic.loc["ml", "tp"] + cic.loc["ml", "fn"])

    cards = "".join([
        card(f"{int(hikari.loc['suricata', 'tp'])} / {hikari_attacks:,}", "encrypted attacks Suricata flagged",
             f"HIKARI-2021 (HTTPS). The model flagged {int(hikari.loc['ml', 'tp']):,}."),
        card(f"0 vs {percent(lab_dec)}", "Suricata, encrypted vs decrypted",
             f"TLS lab, {n_attack:,} attack flows. Decrypting first is what lets the rules see the attacks."),
        card(f"{int(removed['suricata_false_alerts']):,} &rarr; {int(removed['suricata_false_alerts'] - removed['false_alerts_removed'])}",
             "Suricata false alerts after the model filter",
             f"CSE-CIC-IDS2018. {int(removed['suricata_true_alerts'] - removed['true_alerts_lost'])} of "
             f"{int(removed['suricata_true_alerts'])} true alerts kept."),
    ])

    lab_table = lab.assign(**{
        "Flows": lab["flows"].map("{:,}".format),
        "Suricata, encrypted": lab["suricata_encrypted"].map(percent),
        "Suricata, decrypted": lab["suricata_decrypted"].map(percent),
        "HIKARI-trained model": lab["hikari_model"].map(percent),
        "Lab-trained model*": lab["lab_model"].map(percent),
    })[["category", "Flows", "Suricata, encrypted", "Suricata, decrypted", "HIKARI-trained model", "Lab-trained model*"]]
    lab_table = lab_table.rename(columns={"category": "Traffic"})

    per_category = pd.read_csv(ROOT / "integration_results_cic" / "per_category.csv")
    per_category = per_category[per_category["traffic_category"] != "Benign"].assign(
        Flows=lambda d: d["flows"].map("{:,}".format),
        Suricata=lambda d: d["suricata_flagged"].map(percent),
        Model=lambda d: d["ml_flagged"].map(percent),
    )[["traffic_category", "Flows", "Suricata", "Model"]].rename(columns={"traffic_category": "Attack"})

    lab_png = LAB / "lab_detection.png"
    lab_figure = (f'<figure><img src="{image(lab_png)}" alt="TLS lab detection"><figcaption>Share of flows flagged, per traffic type'
                  "</figcaption></figure>") if lab_png.exists() else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Suricata and an ML flow classifier</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; --accent:#2a78d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; --accent:#3987e5; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
main {{ max-width:1040px; margin:0 auto; padding:40px 16px 80px; }}
h1 {{ font-size:30px; margin:0 0 6px; }} h2 {{ font-size:22px; margin:56px 0 4px; }}
.sub {{ color:var(--ink2); margin:0 0 28px; }} .note {{ color:var(--ink2); font-size:14px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:18px; }}
.num {{ font-size:32px; font-weight:700; letter-spacing:-.5px; }} .lab {{ font-weight:600; margin-top:2px; }} .det {{ color:var(--ink2); font-size:14px; margin-top:6px; }}
figure {{ margin:18px 0; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px; }}
figure img {{ width:100%; height:auto; display:block; border-radius:6px; }} figcaption {{ color:var(--ink2); font-size:14px; padding:8px 4px 0; }}
table {{ width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--line); border-radius:12px; font-size:14px; overflow:hidden; }}
th,td {{ padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); }} th {{ color:var(--ink2); font-weight:600; }}
tr:last-child td {{ border-bottom:0; }} td:not(:first-child),th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
code {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:1px 6px; font-size:13px; }}
.caveat {{ border-left:3px solid var(--accent); padding:2px 14px; color:var(--ink2); margin:16px 0; }}
</style></head><body><main>
<h1>Suricata and an ML flow classifier</h1>
<p class="sub">Which attacks each one sees, and what happens when the model filters Suricata's alerts.
All results are offline replays of recorded captures.</p>
<div class="cards">{cards}</div>

<h2>1. HIKARI-2021: encrypted attacks</h2>
<p>The attacks are HTTPS brute force and scans on port 443. Suricata's rules match on content, and with TLS there is none to match.
The model uses packet counts, sizes and timing, which encryption leaves intact.</p>
{figure("01_hikari_suricata_vs_model.png", "Share of attack flows flagged, Suricata vs the model")}
{figure("02_hikari_why_zero.png", "Why Suricata flags nothing here")}
<div class="caveat">The model's precision on HIKARI is only {percent(hikari.loc['ml', 'precision'])}: it also flags {int(hikari.loc['ml', 'fp']):,} benign flows.
<code>check_duplicates.py</code> lists attack flows that also appear as Benign with identical features, which limits how precise any model can be here.</div>

<h2>2. TLS inspection lab: the same attacks encrypted and decrypted</h2>
<p>We generated HIKARI-style traffic against a WordPress site behind an nginx TLS terminator and recorded both the encrypted side
and the plaintext hop behind it. Suricata sees nothing on the encrypted side and most of the attacks once the traffic is decrypted.</p>
{lab_figure}
{table(lab_table)}
<p class="note">* Scored on its own held-out 20% of the flows; the other columns use all flows. The model trained on HIKARI-2021 flags almost nothing
in the lab, so a model has to be trained on the traffic of the network it protects.</p>

<h2>3. CSE-CIC-IDS2018: plaintext attacks</h2>
<p>Here the attacks are mostly readable, so Suricata is back in play. It catches payload attacks such as XSS; the model catches
volumetric and slow attacks such as DoS and brute force.</p>
{figure("04_cic_suricata_vs_model.png", "Share of attack flows flagged, per attack type")}
{table(per_category)}
{figure("05_cic_overlap.png", "Where the two detectors agree and disagree")}

<h2>4. The model as a filter behind Suricata</h2>
<p>An alert is kept only if the model also flags the flow. On the CSE-CIC-IDS2018 held-out flows this removes
{int(removed['false_alerts_removed']):,} of Suricata's {int(removed['suricata_false_alerts']):,} false alerts and loses
{int(removed['true_alerts_lost'])} of its {int(removed['suricata_true_alerts'])} true alerts.</p>
{figure("07_cross_dataset.png", "A model trained on one dataset does not carry over to another")}
{figure("06_cic_unseen_attacks.png", "Attack types the model was not trained on")}

<p class="note">Offline replays of recorded captures; real-time load, memory and packet drops were not measured.
CSE-CIC-IDS2018 labels come from documented attacker addresses, not per-flow ground truth.</p>
</main></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
    if "--open" in sys.argv:
        webbrowser.open(OUT.as_uri())
