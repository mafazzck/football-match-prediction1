"""Football Match Outcome Prediction using Decision Tree and Random Forest.

Select results.csv when prompted, or pass its path as a command-line argument.
All figures and result files are saved in a football_outputs folder beside the CSV.
"""

import os
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_fscore_support
)

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["axes.grid"] = True

# ---------------------------------------------------------
# 1. Select and load the football results CSV
# ---------------------------------------------------------
if len(sys.argv) > 1:
    csv_path = sys.argv[1].strip('"')
else:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askopenfilename(
            title="Select your football results.csv file",
            filetypes=[("CSV files", "*.csv")]
        )
        root.destroy()
    except Exception:
        csv_path = input("Enter the full path to results.csv: ").strip().strip('"')

if not csv_path or not os.path.isfile(csv_path):
    raise FileNotFoundError("Please select a valid football results CSV file.")

output_folder = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "football_outputs")
figures_folder = os.path.join(output_folder, "figures")
os.makedirs(figures_folder, exist_ok=True)

# Set to True if you also want each figure to open in a separate window.
SHOW_PLOTS = False

def save_plot(filename):
    """Save the current plot and optionally display it."""
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, filename), dpi=160, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


df = pd.read_csv(csv_path)
required_columns = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "country", "neutral"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\nFound columns: {list(df.columns)}"
    )

print("Dataset loaded successfully")
print("File:", csv_path)
print("Original shape:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nMissing values before cleaning:\n", df.isnull().sum())

# ---------------------------------------------------------
# 2. Clean data and create the target class
# ---------------------------------------------------------
original_rows = len(df)
original_duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
for col in ["home_team", "away_team", "tournament", "country"]:
    df[col] = df[col].astype("string").str.strip()
    df[col] = df[col].replace("", pd.NA)

before_missing_drop = len(df)
df = df.dropna(subset=required_columns[:-1]).copy()
rows_removed_missing = before_missing_drop - len(df)
df["neutral"] = df["neutral"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0,
                                         "True": 1, "False": 0, 1: 1, 0: 0})
df["neutral"] = pd.to_numeric(df["neutral"], errors="coerce").fillna(0).astype(int)
df = df.sort_values("date", kind="stable").reset_index(drop=True)

def get_result(home_score, away_score):
    if home_score > away_score:
        return "Home Win"
    if home_score < away_score:
        return "Away Win"
    return "Draw"

df["result"] = [get_result(h, a) for h, a in zip(df["home_score"], df["away_score"])]
print("\nRows removed as duplicates:", original_duplicates)
print("Rows removed for missing required data:", rows_removed_missing)
print("Cleaned shape:", df.shape)
print("\nOutcome counts:\n", df["result"].value_counts())

# ---------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
# Figure 1: outcome distribution
plt.figure(figsize=(7, 5))
order = ["Home Win", "Draw", "Away Win"]
df["result"].value_counts().reindex(order, fill_value=0).plot(kind="bar")
plt.title("Football Match Outcome Distribution")
plt.xlabel("Match outcome")
plt.ylabel("Number of matches")
save_plot("01_outcome_distribution.png")

# Figure 2: home and away goal distributions
plt.figure(figsize=(8, 5))
max_goals = int(max(df["home_score"].max(), df["away_score"].max()))
bins = range(0, min(max_goals, 15) + 2)
plt.hist(df["home_score"].clip(upper=15), bins=bins, label="Home goals", alpha=.55)
plt.hist(df["away_score"].clip(upper=15), bins=bins, label="Away goals", alpha=.55)
plt.title("Distribution of Goals (scores above 15 grouped at 15)")
plt.xlabel("Goals scored")
plt.ylabel("Number of matches")
plt.legend()
save_plot("02_goal_distribution.png")

# Figure 3: outcome share by venue type
venue_df = df.copy()
venue_df["venue_type"] = venue_df["neutral"].map({0: "Home venue", 1: "Neutral venue"})
venue_share = pd.crosstab(venue_df["venue_type"], venue_df["result"], normalize="index")
venue_share = venue_share.reindex(columns=order, fill_value=0)
venue_share.plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Match Outcome Share by Venue Type")
plt.xlabel("Venue type")
plt.ylabel("Share of matches")
plt.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
save_plot("03_outcome_by_venue.png")

# Figure 4: ten teams with most wins (home or away)
home_wins = df.loc[df["result"] == "Home Win", "home_team"].value_counts()
away_wins = df.loc[df["result"] == "Away Win", "away_team"].value_counts()
team_wins = home_wins.add(away_wins, fill_value=0).sort_values(ascending=False).head(10).sort_values()
plt.figure(figsize=(9, 6))
team_wins.plot(kind="barh")
plt.title("Top 10 Teams by Total Wins")
plt.xlabel("Number of wins")
plt.ylabel("Team")
save_plot("04_top_teams_by_wins.png")

# Figure 5: matches per decade
matches_by_decade = (df.assign(decade=(df["date"].dt.year // 10) * 10)
                     .groupby("decade").size())
plt.figure(figsize=(11, 5))
matches_by_decade.plot(kind="line", marker="o")
plt.title("International Football Matches by Decade")
plt.xlabel("Decade starting year")
plt.ylabel("Number of matches")
save_plot("05_matches_per_decade.png")

# Figure 6: outcome share for the ten most common tournaments
top_tournaments = df["tournament"].value_counts().head(10).index
tournament_share = pd.crosstab(
    df.loc[df["tournament"].isin(top_tournaments), "tournament"],
    df.loc[df["tournament"].isin(top_tournaments), "result"], normalize="index"
).reindex(columns=order, fill_value=0)
tournament_share = tournament_share.sort_values("Home Win")
tournament_share.plot(kind="barh", stacked=True, figsize=(10, 7))
plt.title("Outcome Share by Tournament (10 Most Common)")
plt.xlabel("Share of matches")
plt.ylabel("Tournament")
plt.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
save_plot("06_outcome_by_tournament.png")

# ---------------------------------------------------------
# 4. Build previous-five-match form features without leakage
# ---------------------------------------------------------
team_history = {}
feature_rows = []

def form_summary(history):
    recent = history[-5:]
    return {
        "matches": len(recent),
        "points": sum(item["points"] for item in recent),
        "wins": sum(item["outcome"] == "W" for item in recent),
        "draws": sum(item["outcome"] == "D" for item in recent),
        "losses": sum(item["outcome"] == "L" for item in recent),
        "goals_for": sum(item["gf"] for item in recent),
        "goals_against": sum(item["ga"] for item in recent),
    }

for match_date, day_matches in df.groupby("date", sort=True):
    day_features = []
    # Calculate every feature for the date before adding any result from that date.
    for _, row in day_matches.iterrows():
        home, away = row["home_team"], row["away_team"]
        home_form = form_summary(team_history.get(home, []))
        away_form = form_summary(team_history.get(away, []))
        features = {
            "date": match_date, "home_team": home, "away_team": away,
            "tournament": row["tournament"], "country": row["country"],
            "neutral": int(row["neutral"]), "year": match_date.year,
            "month": match_date.month,
        }
        for key, value in home_form.items():
            features["home_" + key] = value
        for key, value in away_form.items():
            features["away_" + key] = value
        day_features.append((row, features))

    for row, features in day_features:
        home, away = row["home_team"], row["away_team"]
        hg, ag = int(row["home_score"]), int(row["away_score"])
        if hg > ag:
            ho, ao, hp, ap = "W", "L", 3, 0
        elif hg < ag:
            ho, ao, hp, ap = "L", "W", 0, 3
        else:
            ho, ao, hp, ap = "D", "D", 1, 1
        team_history.setdefault(home, []).append({"outcome": ho, "points": hp, "gf": hg, "ga": ag})
        team_history.setdefault(away, []).append({"outcome": ao, "points": ap, "gf": ag, "ga": hg})
        features["result"] = get_result(hg, ag)
        feature_rows.append(features)

model_df = pd.DataFrame(feature_rows).sort_values("date", kind="stable").reset_index(drop=True)
model_df["form_difference"] = model_df["home_points"] - model_df["away_points"]
model_df.to_csv(os.path.join(output_folder, "football_features.csv"), index=False)
print("\nFeature dataset shape:", model_df.shape)
print("Previous-five-match team form features created.")

# Figure 7: outcome share by difference in recent form
model_df["form_group"] = pd.cut(
    model_df["form_difference"],
    bins=[-float("inf"), -5, -2, 2, 5, float("inf")],
    labels=["Home much weaker", "Home weaker", "Similar form", "Home stronger", "Home much stronger"],
    include_lowest=True
)
form_share = pd.crosstab(model_df["form_group"], model_df["result"], normalize="index").reindex(columns=order, fill_value=0)
form_share.plot(kind="bar", stacked=True, figsize=(10, 5))
plt.title("Outcome Share by Difference in Recent Form")
plt.xlabel("Home points minus away points in previous five matches")
plt.ylabel("Share of matches")
plt.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
save_plot("07_outcome_by_recent_form.png")

# ---------------------------------------------------------
# 5. Select features and split chronologically
# ---------------------------------------------------------
categorical = ["home_team", "away_team", "tournament", "country"]
numerical = [
    "neutral", "year", "month",
    "home_matches", "home_points", "home_wins", "home_draws", "home_losses",
    "home_goals_for", "home_goals_against",
    "away_matches", "away_points", "away_wins", "away_draws", "away_losses",
    "away_goals_for", "away_goals_against",
]
X = model_df[categorical + numerical].copy()
y = model_df["result"].copy()
split_index = int(len(model_df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("\nTest-period outcome distribution:\n", y_test.value_counts())

preprocessor = ColumnTransformer([
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
    ("numerical", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numerical),
])

# ---------------------------------------------------------
# 6. Train models and compare against a simple baseline
# ---------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=11),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=18, min_samples_leaf=5,
        random_state=11, n_jobs=-1
    ),
}
class_order = ["Away Win", "Draw", "Home Win"]
results = []
trained_pipelines = {}

# Report's baseline: predict Home Win for every test match.
baseline_predictions = ["Home Win"] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
results.append({
    "Model": "Baseline (Always Home Win)", "Accuracy": baseline_accuracy,
    "Precision_macro": precision_recall_fscore_support(
        y_test, baseline_predictions, labels=class_order, average="macro", zero_division=0
    )[0],
    "Recall_macro": precision_recall_fscore_support(
        y_test, baseline_predictions, labels=class_order, average="macro", zero_division=0
    )[1],
    "F1_macro": precision_recall_fscore_support(
        y_test, baseline_predictions, labels=class_order, average="macro", zero_division=0
    )[2],
})
print(f"\nBaseline (always Home Win) accuracy: {baseline_accuracy * 100:.2f}%")

for name, model in models.items():
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    trained_pipelines[name] = pipeline
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test, predictions, labels=class_order, output_dict=True, zero_division=0
    )
    results.append({
        "Model": name, "Accuracy": accuracy,
        "Precision_macro": report["macro avg"]["precision"],
        "Recall_macro": report["macro avg"]["recall"],
        "F1_macro": report["macro avg"]["f1-score"],
    })
    print(f"\n{name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions, labels=class_order, zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=class_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_order)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=20)
    plt.title(f"{name} - Confusion Matrix")
    save_plot(f"08_{name.lower().replace(' ', '_')}_confusion_matrix.png")
    pd.DataFrame(cm, index=[f"Actual {c}" for c in class_order], columns=[f"Predicted {c}" for c in class_order]).to_csv(
        os.path.join(output_folder, f"{name.lower().replace(' ', '_')}_confusion_matrix.csv")
    )

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(os.path.join(output_folder, "model_comparison.csv"), index=False)
print("\nModel comparison (including baseline):\n", results_df.to_string(index=False))

# Figure 11: model accuracy comparison, including baseline
plot_results = results_df.sort_values("Accuracy")
plt.figure(figsize=(9, 5))
plt.barh(plot_results["Model"], plot_results["Accuracy"])
plt.xlim(0, min(1, max(.6, float(plot_results["Accuracy"].max()) + .08)))
plt.title("Model Accuracy Comparison")
plt.xlabel("Test accuracy")
plt.ylabel("Model")
save_plot("11_model_accuracy_comparison.png")

# Figure 10: top 10 Random Forest feature importances
rf_pipeline = trained_pipelines["Random Forest"]
rf_model = rf_pipeline.named_steps["model"]
cat_encoder = rf_pipeline.named_steps["preprocessor"].named_transformers_["categorical"].named_steps["encoder"]
cat_names = list(cat_encoder.get_feature_names_out(categorical))
feature_names = cat_names + numerical
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df.to_csv(os.path.join(output_folder, "random_forest_feature_importance.csv"), index=False)
top_importance = importance_df.head(10).sort_values("Importance")
plt.figure(figsize=(9, 6))
plt.barh(top_importance["Feature"], top_importance["Importance"])
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
save_plot("10_random_forest_feature_importance.png")

# ---------------------------------------------------------
# 7. Summary file and one sample prediction
# ---------------------------------------------------------
best_name = results_df[results_df["Model"].isin(models.keys())].iloc[0]["Model"]
best_pipeline = trained_pipelines[best_name]
sample = X_test.iloc[[0]]
actual_result = y_test.iloc[0]
predicted_result = best_pipeline.predict(sample)[0]
print("\nSample prediction")
print("Home team:", sample["home_team"].iloc[0])
print("Away team:", sample["away_team"].iloc[0])
print("Actual outcome:", actual_result)
print("Predicted outcome:", predicted_result)

with open(os.path.join(output_folder, "results_summary.txt"), "w", encoding="utf-8") as file:
    file.write("FOOTBALL MATCH OUTCOME PREDICTION - RESULTS SUMMARY\n")
    file.write(f"Input file: {os.path.basename(csv_path)}\n")
    file.write(f"Original rows: {original_rows}\n")
    file.write(f"Duplicate rows removed: {original_duplicates}\n")
    file.write(f"Rows removed for missing required values: {rows_removed_missing}\n")
    file.write(f"Cleaned rows: {len(df)}\n")
    file.write(f"Training rows: {len(X_train)}\nTesting rows: {len(X_test)}\n")
    file.write(f"Baseline always-Home-Win accuracy: {baseline_accuracy:.4f}\n\n")
    file.write(results_df.to_string(index=False))
    file.write(f"\n\nBest model: {best_name}\n")
    file.write(f"Sample match: {sample['home_team'].iloc[0]} vs {sample['away_team'].iloc[0]}\n")
    file.write(f"Actual result: {actual_result}\nPredicted result: {predicted_result}\n")

print("\nAll output files saved in:", output_folder)
print("Figures saved in:", figures_folder)
print("Note: Results may differ from the report if the dataset or preprocessing differs.")
