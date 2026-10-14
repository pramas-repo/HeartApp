
import os
import pickle
import pandas as pd
import numpy as np

from flask import Flask, render_template, request, flash
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static", "images")

app = Flask(__name__)
app.secret_key = "change-this-development-secret"


def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        return None

    with open(path, "rb") as file:
        return pickle.load(file)


def load_dataset():
    path = os.path.join(BASE_DIR, "heart.csv")

    if os.path.exists(path):
        return pd.read_csv(path)

    return None


def get_features(df):
    return [col for col in df.columns if col != "target"]


@app.route("/")
def dashboard():
    df = load_dataset()

    stats = {
        "records": len(df) if df is not None else 0,
        "features": len(get_features(df)) if df is not None else 0,
        "columns": len(df.columns) if df is not None else 0,
        "missing": int(df.isnull().sum().sum())
        if df is not None else 0
    }

    distribution = {"labels": [], "values": []}
    feature_labels = []
    feature_means = []

    if df is not None and "target" in df.columns:
        counts = df["target"].value_counts().sort_index()
        distribution = {
            "labels": [str(x) for x in counts.index],
            "values": [int(x) for x in counts.values]
        }

        numeric = df.select_dtypes(include="number")
        numeric = numeric.drop(columns=["target"], errors="ignore")
        means = numeric.mean().head(8)

        feature_labels = means.index.tolist()
        feature_means = [
            round(float(x), 2) for x in means.values
        ]

    model_ready = (
        load_pickle("knn.pkl") is not None
        and load_pickle("scaler.pkl") is not None
    )

    image_files = [
        ("Model Accuracy Comparison",
         "model_accuracy_comparison.png"),
        ("Logistic Regression",
         "confusion_matrix_logistic_regression.png"),
        ("Decision Tree",
         "confusion_matrix_decision_tree.png"),
        ("Random Forest",
         "confusion_matrix_random_forest.png"),
        ("SVM", "confusion_matrix_svm.png"),
        ("KNN", "confusion_matrix_knn.png")
    ]

    images = [
        {"title": title, "file": filename}
        for title, filename in image_files
        if os.path.isfile(os.path.join(IMAGE_DIR, filename))
    ]

    return render_template(
        "dashboard.html",
        active="dashboard",
        stats=stats,
        distribution=distribution,
        feature_labels=feature_labels,
        feature_means=feature_means,
        images=images,
        model_ready=model_ready,
        dataset_ready=df is not None
    )


@app.route("/dataset")
def dataset():
    df = load_dataset()

    if df is None:
        flash("Place heart.csv beside app.py.", "warning")
        return render_template(
            "dataset.html",
            active="dataset",
            columns=[],
            rows=[],
            shape=None,
            missing={}
        )

    return render_template(
        "dataset.html",
        active="dataset",
        columns=df.columns.tolist(),
        rows=df.head(15).fillna("").to_dict("records"),
        shape=df.shape,
        missing={
            str(col): int(count)
            for col, count in df.isnull().sum().items()
        }
    )


@app.route("/models")
def models():
    model_images = [
        ("Logistic Regression",
         "confusion_matrix_logistic_regression.png"),
        ("Decision Tree",
         "confusion_matrix_decision_tree.png"),
        ("Random Forest",
         "confusion_matrix_random_forest.png"),
        ("SVM", "confusion_matrix_svm.png"),
        ("KNN", "confusion_matrix_knn.png")
    ]

    images = [
        {"title": title, "file": filename}
        for title, filename in model_images
        if os.path.isfile(os.path.join(IMAGE_DIR, filename))
    ]

    comparison_exists = os.path.isfile(
        os.path.join(
            IMAGE_DIR, "model_accuracy_comparison.png"
        )
    )

    return render_template(
        "models.html",
        active="models",
        images=images,
        comparison_exists=comparison_exists
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    df = load_dataset()

    if df is None:
        flash("heart.csv was not found.", "danger")
        return render_template(
            "predict.html",
            active="predict",
            features=[],
            prediction=None,
            error=None,
            model_ready=False
        )

    features = get_features(df)
    values = {}
    prediction = None
    error = None

    model = load_pickle("knn.pkl")
    scaler = load_pickle("scaler.pkl")
    model_ready = model is not None and scaler is not None

    if request.method == "POST":
        try:
            if not model_ready:
                raise ValueError(
                    "Missing knn.pkl or scaler.pkl. "
                    "Train the model and save both files."
                )

            row = []

            for feature in features:
                raw = request.form.get(feature, "").strip()

                if not raw:
                    raise ValueError(
                        f"Please enter a value for {feature}."
                    )

                value = float(raw)

                if not np.isfinite(value):
                    raise ValueError(
                        f"Invalid numeric value for {feature}."
                    )

                row.append(value)
                values[feature] = raw

            # Preserve the training feature names and order.
            input_df = pd.DataFrame([row], columns=features)

            # Use the SAME scaler fitted during training.
            scaled_input = scaler.transform(input_df)

            result = int(model.predict(scaled_input)[0])

            prediction = {
                "class": result,
                "label": (
                    "Positive classification"
                    if result == 1
                    else "Negative classification"
                ),
                "positive": result == 1
            }

        except Exception as exc:
            error = str(exc)

    feature_info = []

    for feature in features:
        series = pd.to_numeric(
            df[feature], errors="coerce"
        ).dropna()

        feature_info.append({
            "name": feature,
            "min": float(series.min()) if not series.empty else 0,
            "max": float(series.max()) if not series.empty else 100,
            "median": float(series.median()) if not series.empty else 0
        })

    return render_template(
        "predict.html",
        active="predict",
        features=feature_info,
        values=values,
        prediction=prediction,
        error=error,
        model_ready=model_ready
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)