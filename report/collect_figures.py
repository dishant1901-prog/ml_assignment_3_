import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
FIGS = {
    "Q1": ["q1a_curves", "q1b_memorisation"],
    "Q2": ["q2_accuracy_by_condition", "q2_confidence"],
    "Q3": ["q3_reconstructions", "q3_recon_vs_probe"],
    "Q4": ["q4_sweep", "q4_contingency", "q4b_pair_cat_dog", "q4b_pair_automobile_truck", "q4b_pair_bird_deer"],
}

os.makedirs(OUT, exist_ok=True)
for q, names in FIGS.items():
    for n in names:
        for ext in (".pdf", ".png"):
            src = os.path.join(ROOT, q, "figures", n + ext)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(OUT, n + ext))
                print("copied", q, n + ext)
                break
        else:
            print("MISSING", q, n)

