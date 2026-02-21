# ============================================================
# DSCD 611 Final Project
# Global Analysis of Acute Food Insecurity (IPC Phase Classification)
# ============================================================

# -----------------------------
# 1. Library Imports
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 12


# -----------------------------
# 2. Data Loading
# -----------------------------
file_path = r"C:\Users\GIDI\Desktop\MSC DATA SCIENCE\FIRST SEMESTER\DSCD 601 Programming for Data Scientists I\Project\Datasets\raw\IPC_IPC_PHASE.csv"
df_raw = pd.read_csv(file_path)


# -----------------------------
# 3. Data Cleaning & Preparation
# -----------------------------
df_raw["date"] = pd.to_datetime(df_raw["TIME_PERIOD"], format="%Y-%m", errors="coerce")
df_raw["phase"] = df_raw["COMP_BREAKDOWN_2"].str.extract(r"PHASE(\d)").astype(float)

df = df_raw.rename(columns={
    "REF_AREA": "iso3",
    "REF_AREA_LABEL": "country",
    "UNIT_MEASURE": "unit",
    "OBS_VALUE": "value"
})[["iso3", "country", "date", "phase", "unit", "value"]]

df = df.dropna(subset=["date", "phase", "value"])
df["phase"] = df["phase"].astype(int)


# -----------------------------
# 4. Region Definition
# -----------------------------
west_africa = [
    "Benin","Burkina Faso","Cabo Verde","Côte d'Ivoire","Gambia",
    "Ghana","Guinea","Guinea-Bissau","Liberia","Mali","Mauritania",
    "Niger","Nigeria","Senegal","Sierra Leone","Togo","Chad"
]

east_africa = [
    "Burundi","Djibouti","Eritrea","Ethiopia","Kenya","Rwanda",
    "South Sudan","Sudan","Uganda","Tanzania"
]

def assign_region(country):
    if country in west_africa:
        return "West Africa"
    elif country in east_africa:
        return "East Africa"
    else:
        return "Other"

df["Region"] = df["country"].apply(assign_region)


# -----------------------------
# 5. Split Counts vs Percentages
# -----------------------------
df_people = df[df["unit"] == "PS"].copy()
df_pct = df[df["unit"] == "PT"].copy()


# -----------------------------
# 6. Pivot to Wide Format
# -----------------------------
wide_people = df_people.pivot_table(
    index=["iso3", "country", "Region", "date"],
    columns="phase",
    values="value",
    aggfunc="sum"
).reset_index()

wide_pct = df_pct.pivot_table(
    index=["iso3", "country", "Region", "date"],
    columns="phase",
    values="value",
    aggfunc="mean"
).reset_index()

wide_people.columns = [f"phase_{int(c)}_people" if isinstance(c, int) else c for c in wide_people.columns]
wide_pct.columns = [f"phase_{int(c)}_pct" if isinstance(c, int) else c for c in wide_pct.columns]

wide_people["crisis_plus_people"] = (
    wide_people[["phase_3_people","phase_4_people","phase_5_people"]]
    .fillna(0).sum(axis=1)
)

wide_pct["crisis_plus_pct"] = (
    wide_pct[["phase_3_pct","phase_4_pct","phase_5_pct"]]
    .fillna(0).sum(axis=1)
)


# ============================================================
# QUESTION 1: Global Trend (%)
# ============================================================
global_trend = wide_pct.groupby("date")["crisis_plus_pct"].mean().reset_index()

plt.plot(global_trend["date"], global_trend["crisis_plus_pct"], linewidth=2.5)
plt.title("Global Average % of Population in Phase 3+")
plt.ylabel("Percentage (%)")
plt.xlabel("Date")
plt.show()


# ============================================================
# QUESTION 2: Global Burden Contribution (People Share)
# ============================================================
latest_people = wide_people.loc[wide_people.groupby("country")["date"].idxmax()]
latest_people["global_share"] = (
    latest_people["crisis_plus_people"] /
    latest_people["crisis_plus_people"].sum()
) * 100

top_burden = latest_people.sort_values("global_share", ascending=False).head(10)

ax = sns.barplot(data=top_burden, y="country", x="global_share")
for p in ax.patches:
    ax.annotate(
        f"{p.get_width():.1f}%",
        (p.get_width(), p.get_y() + p.get_height()/2),
        ha="left", va="center"
    )

plt.title("Top Contributors to Global Phase 3+ Population")
plt.xlabel("Share of Global Crisis Population (%)")
plt.show()


# ============================================================
# QUESTION 3: Top 5 Countries by % Population in Phase 3+ (Latest)
# ============================================================
latest_pct = wide_pct.loc[wide_pct.groupby("country")["date"].idxmax()]
top5_severity = latest_pct.sort_values("crisis_plus_pct", ascending=False).head(5)

ax = sns.barplot(data=top5_severity, y="country", x="crisis_plus_pct")
for p in ax.patches:
    ax.annotate(
        f"{p.get_width():.1f}%",
        (p.get_width(), p.get_y() + p.get_height()/2),
        ha="left", va="center"
    )

plt.title("Top 5 Countries by % of Population in Phase 3+")
plt.xlabel("Percentage (%)")
plt.show()


# ============================================================
# QUESTION 4: Concentration of Global Crisis
# ============================================================
top5_share = top_burden.head(5)["global_share"].sum()
print(f"Top 5 countries account for {top5_share:.1f}% of the global Phase 3+ population.")


# ============================================================
# QUESTION 5: West vs East Africa (%)
# ============================================================
regional_trend = (
    wide_pct[wide_pct["Region"].isin(["West Africa","East Africa"])]
    .groupby(["Region","date"])["crisis_plus_pct"]
    .mean().reset_index()
)

sns.lineplot(data=regional_trend, x="date", y="crisis_plus_pct", hue="Region")
plt.title("West vs East Africa: Average % in Phase 3+")
plt.ylabel("Percentage (%)")
plt.show()


# ============================================================
# QUESTION 6: Statistical Significance
# ============================================================
west = regional_trend[regional_trend["Region"]=="West Africa"]["crisis_plus_pct"]
east = regional_trend[regional_trend["Region"]=="East Africa"]["crisis_plus_pct"]

t_stat, p_val = ttest_ind(west, east, equal_var=False)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.5f}")


# ============================================================
# QUESTION 7: Depth of Crisis (Phase 4–5 Share)
# ============================================================
wide_pct["severe_share"] = (
    wide_pct[["phase_4_pct","phase_5_pct"]].sum(axis=1) /
    wide_pct["crisis_plus_pct"]
)

depth = (
    wide_pct.groupby("country")["severe_share"]
    .mean()
    .reset_index()
    .sort_values("severe_share", ascending=False)
    .head(10)
)

ax = sns.barplot(data=depth, y="country", x="severe_share")
for p in ax.patches:
    ax.annotate(
        f"{p.get_width():.2f}",
        (p.get_width(), p.get_y() + p.get_height()/2),
        ha="left", va="center"
    )

plt.title("Countries with Deepest Crisis (Phase 4–5 Share of Phase 3+)")
plt.xlabel("Severity Ratio")
plt.show()


# ============================================================
# QUESTION 8: Fastest Deterioration (% Slope)
# ============================================================
slopes = []

for country, grp in wide_pct.groupby("country"):
    if len(grp) > 6:
        grp = grp.sort_values("date")
        X = np.arange(len(grp)).reshape(-1,1)
        y = grp["crisis_plus_pct"].values
        slope = LinearRegression().fit(X, y).coef_[0]
        slopes.append((country, slope))

slope_df = pd.DataFrame(slopes, columns=["country","slope"])
fastest = slope_df.sort_values("slope", ascending=False).head(10)

ax = sns.barplot(data=fastest, y="country", x="slope")
for p in ax.patches:
    ax.annotate(
        f"{p.get_width():.2f}",
        (p.get_width(), p.get_y() + p.get_height()/2),
        ha="left", va="center"
    )

plt.title("Fastest Deteriorating Countries (Increase in Phase 3+ %)")
plt.xlabel("Slope")
plt.show()


# ============================================================
# QUESTION 9: Volatility vs Severity
# ============================================================
stats = wide_pct.groupby("country")["crisis_plus_pct"].agg(["mean","std"]).reset_index()
corr, p_corr = pearsonr(stats["mean"], stats["std"])

print(f"Correlation between severity and volatility: r={corr:.2f}, p={p_corr:.5f}")

sns.scatterplot(data=stats, x="mean", y="std")
plt.title("Volatility vs Average Severity in Phase 3+")
plt.xlabel("Mean % in Phase 3+")
plt.ylabel("Volatility (Std Dev)")
plt.show()


# ============================================================
# QUESTION 10: Recovery vs Persistence
# ============================================================
recovery = slope_df.sort_values("slope").head(10)

print("\nCountries Showing Strongest Recovery (Declining Phase 3+ %):")
print(recovery)
