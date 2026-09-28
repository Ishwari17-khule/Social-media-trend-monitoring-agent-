"""
sentiment_analysis.py
--------------------------------------------------------------------
PURPOSE:
    Trains a sentiment classification model using the Twitter datasets
    (twitter_training.csv for training, twitter_validation.csv for testing).

    PIPELINE:
        1. Clean tweet text (using preprocessing.py)
        2. Convert text to numeric features using TF-IDF
        3. Train a Logistic Regression classifier
        4. Predict sentiments on the validation set
        5. Evaluate using accuracy_score & classification_report
        6. Return everything the dashboard needs to display results
--------------------------------------------------------------------
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import add_clean_text_column


def train_sentiment_model(train_df: pd.DataFrame, val_df: pd.DataFrame, max_features: int = 5000) -> dict:
    """
    Train a Logistic Regression sentiment classifier on the training data
    and evaluate it on the validation data.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must contain 'text' and 'sentiment' columns (twitter_training.csv)
    val_df : pd.DataFrame
        Must contain 'text' and 'sentiment' columns (twitter_validation.csv)
    max_features : int
        Maximum number of TF-IDF features to keep (controls vocabulary size)

    Returns
    -------
    dict with keys:
        - model           : trained LogisticRegression model
        - vectorizer      : fitted TfidfVectorizer (needed to transform new text)
        - accuracy        : float, accuracy on validation set
        - report          : str, full classification report
        - predictions_df  : DataFrame with text, actual & predicted sentiment
    """

    # ----------------------------------------------------------------
    # STEP 1: Apply text preprocessing -> creates a 'clean_text' column
    # ----------------------------------------------------------------
    train_df = add_clean_text_column(train_df, source_col="text")
    val_df = add_clean_text_column(val_df, source_col="text")

    # Remove rows that became empty strings after cleaning
    train_df = train_df[train_df["clean_text"].str.strip() != ""]
    val_df = val_df[val_df["clean_text"].str.strip() != ""]

    X_train_text = train_df["clean_text"]
    y_train = train_df["sentiment"]

    X_val_text = val_df["clean_text"]
    y_val = val_df["sentiment"]

    # ----------------------------------------------------------------
    # STEP 2: TF-IDF Vectorization
    # ----------------------------------------------------------------
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)

    # ----------------------------------------------------------------
    # STEP 3: Train Logistic Regression model
    # (MultinomialNB could be swapped in here if preferred)
    # ----------------------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # ----------------------------------------------------------------
    # STEP 4: Predict sentiments on the validation set
    # ----------------------------------------------------------------
    y_pred = model.predict(X_val)

    # ----------------------------------------------------------------
    # STEP 5: Evaluate the model
    # ----------------------------------------------------------------
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, zero_division=0)
    report_dict = classification_report(y_val, y_pred, zero_division=0, output_dict=True)

    # ----------------------------------------------------------------
    # STEP 6: Package results into a results DataFrame
    # ----------------------------------------------------------------
    predictions_df = pd.DataFrame({
        "text": val_df["text"].values,
        "entity": val_df["entity"].values if "entity" in val_df.columns else "",
        "actual_sentiment": y_val.values,
        "predicted_sentiment": y_pred
    })

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_val)
        predictions_df["confidence"] = probabilities.max(axis=1)
    else:
        predictions_df["confidence"] = 0.0

    labels = sorted(pd.Series(y_val).dropna().unique())
    confusion_df = pd.DataFrame(
        confusion_matrix(y_val, y_pred, labels=labels),
        index=labels,
        columns=labels,
    )

    per_class_df = (
        pd.DataFrame(report_dict)
        .transpose()
        .reset_index()
        .rename(columns={"index": "sentiment"})
    )
    per_class_df = per_class_df[per_class_df["sentiment"].isin(labels)]

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "report": report,
        "report_dict": report_dict,
        "per_class_df": per_class_df,
        "confusion_df": confusion_df,
        "predictions_df": predictions_df
    }


# ==================================================================
# Quick manual test
# ==================================================================
if __name__ == "__main__":
    from src.data_collection import load_twitter_training, load_twitter_validation

    train_data = load_twitter_training()
    val_data = load_twitter_validation()

    results = train_sentiment_model(train_data, val_data)

    print("Accuracy:", results["accuracy"])
    print(results["report"])
    print(results["predictions_df"].head())
