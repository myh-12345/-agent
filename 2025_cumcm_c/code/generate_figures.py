import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(r"E:/下载/SvpohSGacdffe718bcaa3b6e835c03ae3461cab1/C题")
DATA_FILE = ROOT / "附件.xlsx"
FIG_DIR = ROOT / "figures"
OUT_STATS = ROOT / "code" / "summary_stats.txt"


def parse_week(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(" ", "")
    match = re.match(r"(\d+)w(?:\+(\d+))?", text)
    if not match:
        return np.nan
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    return weeks + days / 7


def apply_paper_style(width_fraction=0.72, text_width_in=6.3, base_font_pt=11.0, aspect_ratio=0.62):
    fig_width = text_width_in * width_fraction
    fig_height = fig_width * aspect_ratio
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": base_font_pt,
            "axes.titlesize": base_font_pt,
            "axes.labelsize": base_font_pt,
            "xtick.labelsize": base_font_pt - 0.5,
            "ytick.labelsize": base_font_pt - 0.5,
            "legend.fontsize": base_font_pt - 0.5,
            "legend.title_fontsize": base_font_pt - 0.5,
            "savefig.bbox": "tight",
            "figure.dpi": 220,
        }
    )
    return fig_width, fig_height


def load_data():
    male = pd.read_excel(DATA_FILE, sheet_name="男胎检测数据")
    female = pd.read_excel(DATA_FILE, sheet_name="女胎检测数据")
    male["week"] = male["检测孕周"].apply(parse_week)
    female["week"] = female["检测孕周"].apply(parse_week)
    return male, female


def build_mother_earliest(male):
    rows = []
    for code, group in male.sort_values(["孕妇代码", "week"]).groupby("孕妇代码"):
        group = group.sort_values("week")
        reached = group[group["Y染色体浓度"] >= 0.04]
        earliest = reached["week"].min() if len(reached) else np.nan
        rows.append(
            {
                "孕妇代码": code,
                "BMI": group["孕妇BMI"].iloc[0],
                "earliest_week": earliest,
            }
        )
    mother = pd.DataFrame(rows)
    bins = [0, 30, 34, 38, 100]
    labels = ["[0,30)", "[30,34)", "[34,38)", "[38,+inf)"]
    mother["BMI_group"] = pd.cut(mother["BMI"], bins=bins, labels=labels, right=False)
    return mother


def save_dual_format(name):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=220)
    plt.savefig(FIG_DIR / f"{name}.pdf")


def plot_y_vs_week(male):
    fig_w, fig_h = apply_paper_style()
    plt.figure(figsize=(fig_w, fig_h))
    sampled = male.sample(min(len(male), 600), random_state=42)
    sns.scatterplot(
        data=sampled,
        x="week",
        y="Y染色体浓度",
        hue="孕妇BMI",
        palette="viridis",
        s=18,
        edgecolor=None,
        alpha=0.75,
    )
    sns.regplot(
        data=male,
        x="week",
        y="Y染色体浓度",
        scatter=False,
        color="crimson",
        line_kws={"linewidth": 1.6},
    )
    plt.axhline(0.04, color="black", linestyle="--", linewidth=1.2, label="4% threshold")
    plt.xlabel("Gestational week")
    plt.ylabel("Y chromosome concentration")
    plt.title("Y chromosome concentration versus gestational week")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    save_dual_format("y_concentration_vs_week")
    plt.close()


def plot_bmi_reach_curve(mother):
    fig_w, fig_h = apply_paper_style()
    plt.figure(figsize=(fig_w, fig_h))
    weeks = list(range(12, 24))
    for group_name, group in mother.groupby("BMI_group", observed=False):
        if len(group) == 0:
            continue
        values = [float((group["earliest_week"] <= wk).mean()) for wk in weeks]
        plt.plot(weeks, values, marker="o", linewidth=1.8, markersize=4, label=str(group_name))
    plt.xlabel("Gestational week")
    plt.ylabel("Proportion reaching Y concentration >= 4%")
    plt.title("Reach-rate curves under different BMI groups")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    save_dual_format("bmi_group_reach_curve")
    plt.close()


def plot_female_rule(female):
    female = female.copy()
    female["abnormal"] = female["染色体的非整倍体"].notna().astype(int)
    female["zmax"] = female[["13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值"]].max(axis=1)

    fig_w, fig_h = apply_paper_style()
    plt.figure(figsize=(fig_w, fig_h))
    sns.scatterplot(
        data=female,
        x="X染色体浓度",
        y="zmax",
        hue="abnormal",
        palette={0: "#2563eb", 1: "#dc2626"},
        s=16,
        alpha=0.8,
    )
    plt.axvline(-0.015, color="black", linestyle="--", linewidth=1.2, label="X threshold")
    plt.axhline(2.5, color="darkgreen", linestyle="--", linewidth=1.2, label="Z threshold")
    plt.xlabel("X chromosome concentration")
    plt.ylabel("max(Z13, Z18, Z21)")
    plt.title("Female-fetus abnormality decision rule")
    plt.legend(
        title="Class",
        labels=["normal", "abnormal", "X threshold", "Z threshold"],
        loc="upper right",
        frameon=True,
    )
    plt.tight_layout()
    save_dual_format("female_abnormal_rule")
    plt.close()


def save_stats(male, female, mother):
    lines = []
    lines.append(f"male_rows={len(male)}")
    lines.append(f"female_rows={len(female)}")
    lines.append(f"male_mothers={male['孕妇代码'].nunique()}")
    lines.append(f"female_mothers={female['孕妇代码'].nunique()}")
    lines.append("male_corr:")
    corr = male[["week", "孕妇BMI", "年龄", "身高", "体重", "Y染色体浓度"]].corr(numeric_only=True)["Y染色体浓度"]
    for k, v in corr.items():
        lines.append(f"  {k}: {v:.6f}")
    lines.append("bmi_group_summary:")
    group_summary = mother.groupby("BMI_group", observed=False)["earliest_week"].agg(["count", "mean", "median"])
    lines.append(group_summary.to_string())
    OUT_STATS.write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    male, female = load_data()
    mother = build_mother_earliest(male)
    sns.set_theme(style="whitegrid")
    plot_y_vs_week(male)
    plot_bmi_reach_curve(mother)
    plot_female_rule(female)
    save_stats(male, female, mother)


if __name__ == "__main__":
    main()
