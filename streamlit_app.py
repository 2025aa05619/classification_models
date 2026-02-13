from __future__ import annotations
from pathlib import Path
import pickle
from typing import Any
import pandas as pd
import streamlit as st
from model.training_utils import (
    evaluate_model,
    load_artifacts,
    load_dataset_from_csv,
    scale_test_features,
    encode_target,
    run_training_pipeline
)


ARTIFACTS_DIR = Path(__file__).parent / "model" / "artifacts"


def _format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


@st.cache_resource
def _get_trained_models_and_baseline_metrics(random_state: int = 42):
    existing = load_artifacts(ARTIFACTS_DIR)
    if existing is not None:
        print("Loaded existing trained models and metrics from artifacts.")
        models, metrics_df = existing
        return models, metrics_df

    return run_training_pipeline()


def _read_uploaded_csv(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if df.empty:
        raise ValueError("Uploaded CSV is empty.")
    return df


def _plot_confusion_matrix(cm, labels):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig, clear_figure=True)


def main():
    bundle = load_dataset_from_csv(
        Path(__file__).parent / "model" / "dataset_full.csv",
    )
    st.set_page_config(page_title="ML Classification Models Demo", layout="wide")

    st.title("Classification Models: Training, Evaluation, Deployment (Streamlit)")
    st.caption(f"Dataset: **{bundle.name}** (UCI Machine Learning Repository)")
    st.caption("Use **Download sample test CSV** button in the sidebar to get a sample CSV file to evaluate a selected model.")

    trained_models, baseline_metrics_df = _get_trained_models_and_baseline_metrics()

    with st.sidebar:
        st.subheader("Download Sample")
        sample_csv_path = Path(__file__).parent / "model" / "sample_test.csv"
        if sample_csv_path.exists():
            csv_data = sample_csv_path.read_text()
        else:
            # Fallback: generate sample if file doesn't exist
            sample_df = bundle.X.head(100).copy()
            sample_df["target"] = bundle.y.head(100)
            csv_data = sample_df.to_csv(index=False)
        st.download_button(
            label="Download sample test CSV",
            data=csv_data,
            file_name="sample_test.csv",
            mime="text/csv"
        )
        st.divider()

        st.header("Select Model")
        model_name = st.selectbox("Select a model", options=list(trained_models.keys()))
        st.divider()
        st.subheader("Upload test CSV")
        upload = st.file_uploader("Upload CSV (test data)", type=["csv"])
        has_target = st.checkbox("CSV includes target column", value=True)
        target_col = None
        if upload is not None:
            try:
                uploaded_df = _read_uploaded_csv(upload)
                cols = uploaded_df.columns.tolist()
                if has_target:
                    default_target = "target" if "target" in cols else cols[-1]
                    target_col = st.selectbox("Target column", options=cols, index=cols.index(default_target))
            except Exception as e:
                st.error(str(e))
                uploaded_df = None
        else:
            uploaded_df = None

    # Split dataset preview and baseline metrics in one row
    dataset_preview_left, baseline_metrics_right = st.columns(2, gap="medium")
    
    with dataset_preview_left:
        st.subheader("Dataset preview")
        preview = bundle.X.copy()
        target_name = bundle.y.name if bundle.y.name else "target"
        preview[target_name] = bundle.y
        preview.columns = [f"{i+1}. {col}" if col != target_name else f"{target_name} (target)" for i, col in enumerate(preview.columns)]
        st.dataframe(preview.head(10), width='stretch')
    
    with baseline_metrics_right:
        st.subheader("Baseline metrics")
        st.dataframe(baseline_metrics_df.style.format(_format_metric), width='stretch')
    
    # Create a nested column structure for better alignment
    pred_col, eval_col = st.columns(2, gap="medium")
    
    with pred_col:
        st.subheader(f"Selected model: {model_name}")
        model = trained_models[model_name]

        if uploaded_df is None:
            st.info("Upload a CSV to evaluate this model on your test data.")
        else:
            df = uploaded_df.copy()
            if has_target:
                if target_col is None or target_col not in df.columns:
                    st.error("Select a valid target column.")
                    return
                y_test = df[target_col]
                X_test = df.drop(columns=[target_col])
            else:
                y_test = None
                X_test = df

            missing = [c for c in bundle.X.columns if c not in X_test.columns]
            extra = [c for c in X_test.columns if c not in bundle.X.columns]
            if missing:
                st.error(f"Missing required feature columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")
                return
            if extra:
                st.warning(f"Ignoring extra columns not used by the model: {extra[:8]}{'...' if len(extra) > 8 else ''}")
                X_test = X_test[bundle.X.columns]
            else:
                X_test = X_test[bundle.X.columns]

            # Scale features and encode target
            scaler_path = ARTIFACTS_DIR / "scaler.pkl"
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            X_test = scale_test_features(X_test, scaler)
            if y_test is not None:
                encoder_path = ARTIFACTS_DIR / "label_encoder.pkl"
                with open(encoder_path, "rb") as f:
                    label_encoder = pickle.load(f)
                y_test = encode_target(y_test, encoder=label_encoder)[0]

            st.subheader("Predictions")
            y_pred = model.predict(X_test)
            preds_df = pd.DataFrame({"prediction": y_pred})
            st.dataframe(preds_df, width='stretch')

    with eval_col:
        if uploaded_df is not None and y_test is not None:
            try:
                y_test_series = pd.Series(y_test).astype(int)
            except Exception:
                st.error("Target column must be numeric (0/1) for this dataset.")
                return
            
            st.subheader("")
            st.subheader("Evaluation Metrics")
            metrics = evaluate_model(model, X_test, y_test_series)
            # Display metrics in table format
            metrics_table = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                "Value": [
                    metrics["Accuracy"],
                    metrics["AUC"],
                    metrics["Precision"],
                    metrics["Recall"],
                    metrics["F1"],
                    metrics["MCC"]
                ]
            })
            st.dataframe(metrics_table.style.format({"Value": _format_metric}), width='stretch', hide_index=True)
        elif uploaded_df is not None:
            st.info("No target column provided. Showing predictions only.")
    
    # Second row for confusion matrix and classification report
    if uploaded_df is not None and y_test is not None:
        cm_col, report_col = st.columns(2, gap="medium")
        
        with cm_col:
            st.subheader("Confusion Matrix")
            cm = metrics["ConfusionMatrix"]
            _plot_confusion_matrix(cm, labels=["class 0", "class 1"])
        
        with report_col:
            st.subheader("Classification Report")
            st.json(metrics["ClassificationReport"])


if __name__ == "__main__":
    main()
