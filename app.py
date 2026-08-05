from pathlib import Path
import html

import gradio as gr
import joblib
import pandas as pd


# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ExtraTrees.pkl"
DATA_PATH = BASE_DIR / "Raw_Data_Set.xlsx"

IMAGE_PATH = BASE_DIR / "Steel and Aluminium.png"
VIDEO_PATH = BASE_DIR / "Axial Deformation.mp4"


# =========================================================
# MODEL INPUT COLUMNS
# These names must match the columns used during training.
# =========================================================
FEATURE_COLS = [
    "E Fixed",
    "rho Fixed",
    "nu Fixed",
    "E Free",
    "rho Free",
    "nu Free",
]

DISPLAY_NAMES = {
    "E Fixed": "E (Fixed) [N/m²]",
    "rho Fixed": "ρ (Fixed) [kg/m³]",
    "nu Fixed": "ν (Fixed) [-]",
    "E Free": "E (Free) [N/m²]",
    "rho Free": "ρ (Free) [kg/m³]",
    "nu Free": "ν (Free) [-]",
}


# =========================================================
# COLUMN NORMALIZATION
# This is used only for the optional validity-range table.
# =========================================================
def simplify_name(name: str) -> str:
    text = str(name).strip().lower()

    text = text.replace("ρ", "rho")
    text = text.replace("ν", "nu")

    for character in [
        " ",
        "_",
        "-",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        ".",
        "/",
        "\\",
    ]:
        text = text.replace(character, "")

    return text


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    canonical_names = {
        "efixed": "E Fixed",
        "rhofixed": "rho Fixed",
        "nufixed": "nu Fixed",
        "efree": "E Free",
        "rhofree": "rho Free",
        "nufree": "nu Free",
        "axialfrequencyhz": "Axial Frequency (Hz)",
        "axialfrequency": "Axial Frequency (Hz)",
    }

    rename_dictionary = {}

    for original_column in dataframe.columns:
        simplified_column = simplify_name(original_column)

        if simplified_column in canonical_names:
            rename_dictionary[original_column] = (
                canonical_names[simplified_column]
            )

    return dataframe.rename(columns=rename_dictionary)


# =========================================================
# LOAD SAVED MODEL
# The application remains visible even if model loading fails.
# =========================================================
MODEL = None
MODEL_ERROR = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found."
        )

    MODEL = joblib.load(MODEL_PATH)

except Exception as error:
    MODEL_ERROR = str(error)


# =========================================================
# LOAD OPTIONAL REFERENCE DATASET
# =========================================================
REFERENCE_DATA = None
DATA_ERROR = None

try:
    if DATA_PATH.exists():
        reference_data = pd.read_excel(DATA_PATH)
        reference_data = normalize_columns(reference_data)

        missing_columns = [
            column
            for column in FEATURE_COLS
            if column not in reference_data.columns
        ]

        if missing_columns:
            raise ValueError(
                "The following input columns are missing "
                f"from Raw_Data_Set.xlsx: {missing_columns}"
            )

        reference_data = reference_data[FEATURE_COLS].copy()

        for column in FEATURE_COLS:
            reference_data[column] = pd.to_numeric(
                reference_data[column],
                errors="coerce",
            )

        REFERENCE_DATA = (
            reference_data
            .dropna(how="all")
            .reset_index(drop=True)
        )

except Exception as error:
    DATA_ERROR = str(error)


# =========================================================
# INPUT DATAFRAME
# =========================================================
def create_input_dataframe(
    e_fixed,
    rho_fixed,
    nu_fixed,
    e_free,
    rho_free,
    nu_free,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "E Fixed": float(e_fixed),
                "rho Fixed": float(rho_fixed),
                "nu Fixed": float(nu_fixed),
                "E Free": float(e_free),
                "rho Free": float(rho_free),
                "nu Free": float(nu_free),
            }
        ]
    )


# =========================================================
# INPUT VALIDATION
# =========================================================
def validate_inputs(
    e_fixed,
    rho_fixed,
    nu_fixed,
    e_free,
    rho_free,
    nu_free,
) -> None:
    values = [
        e_fixed,
        rho_fixed,
        nu_fixed,
        e_free,
        rho_free,
        nu_free,
    ]

    if any(value is None for value in values):
        raise ValueError(
            "Please enter all six material properties."
        )

    if float(e_fixed) <= 0:
        raise ValueError(
            "E (Fixed) must be greater than zero."
        )

    if float(rho_fixed) <= 0:
        raise ValueError(
            "ρ (Fixed) must be greater than zero."
        )

    if float(e_free) <= 0:
        raise ValueError(
            "E (Free) must be greater than zero."
        )

    if float(rho_free) <= 0:
        raise ValueError(
            "ρ (Free) must be greater than zero."
        )

    if not 0 <= float(nu_fixed) < 0.5:
        raise ValueError(
            "ν (Fixed) must be between 0 and 0.5."
        )

    if not 0 <= float(nu_free) < 0.5:
        raise ValueError(
            "ν (Free) must be between 0 and 0.5."
        )


# =========================================================
# VALIDITY TABLE
# =========================================================
def build_validity_table(
    input_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    rows = []
    outside_count = 0

    for feature in FEATURE_COLS:
        current_value = float(
            input_dataframe.iloc[0][feature]
        )

        if (
            REFERENCE_DATA is not None
            and not REFERENCE_DATA.empty
        ):
            minimum_value = float(
                REFERENCE_DATA[feature].min()
            )

            maximum_value = float(
                REFERENCE_DATA[feature].max()
            )

            inside_range = (
                minimum_value
                <= current_value
                <= maximum_value
            )

            status = (
                "Inside"
                if inside_range
                else "Outside"
            )

            if not inside_range:
                outside_count += 1

            minimum_display = f"{minimum_value:.6e}"
            maximum_display = f"{maximum_value:.6e}"

        else:
            status = "Not checked"
            minimum_display = "N/A"
            maximum_display = "N/A"

        rows.append(
            {
                "Feature": DISPLAY_NAMES[feature],
                "Current Value": f"{current_value:.6e}",
                "Training Minimum": minimum_display,
                "Training Maximum": maximum_display,
                "Status": status,
            }
        )

    return pd.DataFrame(rows), outside_count


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_axial_frequency(
    e_fixed,
    rho_fixed,
    nu_fixed,
    e_free,
    rho_free,
    nu_free,
):
    try:
        if MODEL is None:
            raise RuntimeError(
                "The saved model could not be loaded. "
                f"Details: {MODEL_ERROR}"
            )

        validate_inputs(
            e_fixed=e_fixed,
            rho_fixed=rho_fixed,
            nu_fixed=nu_fixed,
            e_free=e_free,
            rho_free=rho_free,
            nu_free=nu_free,
        )

        input_dataframe = create_input_dataframe(
            e_fixed=e_fixed,
            rho_fixed=rho_fixed,
            nu_fixed=nu_fixed,
            e_free=e_free,
            rho_free=rho_free,
            nu_free=nu_free,
        )

        prediction = float(
            MODEL.predict(
                input_dataframe[FEATURE_COLS]
            )[0]
        )

        validity_table, outside_count = (
            build_validity_table(input_dataframe)
        )

        result_html = f"""
        <div class="prediction-card">
            <div class="prediction-title">
                Predicted Axial Frequency
            </div>

            <div class="prediction-number">
                {prediction:,.2f}
                <span class="prediction-unit">Hz</span>
            </div>

            <div class="prediction-model">
                Prediction model: ExtraTrees Regressor
            </div>
        </div>
        """

        if REFERENCE_DATA is None:
            if DATA_ERROR:
                status_message = (
                    "ℹ️ Prediction completed. The training-range "
                    f"check was unavailable: `{DATA_ERROR}`"
                )
            else:
                status_message = (
                    "ℹ️ Prediction completed. "
                    "`Raw_Data_Set.xlsx` was not included, "
                    "so the training-range check was skipped."
                )

        elif outside_count > 0:
            status_message = (
                f"⚠️ **Warning:** {outside_count} input value(s) "
                "are outside the training-data range. "
                "This prediction involves extrapolation and "
                "may be less reliable."
            )

        else:
            status_message = (
                "✅ All input values are inside the "
                "training-data range."
            )

        return (
            result_html,
            status_message,
            validity_table,
        )

    except Exception as error:
        escaped_error = html.escape(str(error))

        error_html = f"""
        <div class="error-card">
            <div class="error-title">
                Prediction could not be completed
            </div>

            <div class="error-message">
                {escaped_error}
            </div>
        </div>
        """

        empty_table = pd.DataFrame(
            columns=[
                "Feature",
                "Current Value",
                "Training Minimum",
                "Training Maximum",
                "Status",
            ]
        )

        return (
            error_html,
            "Please correct the inputs or check the Space logs.",
            empty_table,
        )


# =========================================================
# CUSTOM CSS
# =========================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.app-subtitle {
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 1.25rem;
}

.prediction-card {
    background: linear-gradient(135deg, #0f5132, #198754);
    border-radius: 18px;
    padding: 30px;
    margin-top: 15px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
}

.prediction-title {
    color: #d9fbe5;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 14px;
}

.prediction-number {
    color: white;
    font-size: 3rem;
    font-weight: 850;
    line-height: 1.1;
}

.prediction-unit {
    color: #d9fbe5;
    font-size: 1.3rem;
    font-weight: 650;
}

.prediction-model {
    color: #d9fbe5;
    margin-top: 14px;
    font-size: 0.95rem;
}

.error-card {
    background: rgba(220, 38, 38, 0.12);
    border: 1px solid rgba(220, 38, 38, 0.45);
    border-radius: 14px;
    padding: 20px;
    margin-top: 15px;
}

.error-title {
    color: #dc2626;
    font-size: 1.2rem;
    font-weight: 750;
    margin-bottom: 10px;
}

.error-message {
    font-size: 1rem;
}
"""


# =========================================================
# GRADIO GUI
# =========================================================
with gr.Blocks(
    title="Axial Frequency Predictor",
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown(
        """
        # ⚙️ Axial Frequency Predictor

        <div class="app-subtitle">
        Predict axial frequency using a trained
        ExtraTrees regression model.
        Enter the fixed-material and free-material
        properties below.
        </div>
        """
    )

    if IMAGE_PATH.exists():
        gr.Image(
            value=str(IMAGE_PATH),
            show_label=False,
            interactive=False,
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Fixed Material")

            e_fixed_input = gr.Number(
                label="E (Fixed) [N/m²]",
                value=1.97e11,
            )

            rho_fixed_input = gr.Number(
                label="ρ (Fixed) [kg/m³]",
                value=7750.3,
            )

            nu_fixed_input = gr.Number(
                label="ν (Fixed) [-]",
                value=0.29,
            )

        with gr.Column():
            gr.Markdown("## Free Material")

            e_free_input = gr.Number(
                label="E (Free) [N/m²]",
                value=4.24e8,
            )

            rho_free_input = gr.Number(
                label="ρ (Free) [kg/m³]",
                value=2200.5,
            )

            nu_free_input = gr.Number(
                label="ν (Free) [-]",
                value=0.45,
            )

    predict_button = gr.Button(
        "Predict Axial Frequency",
        variant="primary",
    )

    result_output = gr.HTML()

    status_output = gr.Markdown()

    with gr.Accordion(
        "Training Validity Region",
        open=False,
    ):
        validity_output = gr.Dataframe(
            headers=[
                "Feature",
                "Current Value",
                "Training Minimum",
                "Training Maximum",
                "Status",
            ],
            interactive=False,
        )

    if VIDEO_PATH.exists():
        gr.Markdown("## Axial Deformation")

        gr.Video(
            value=str(VIDEO_PATH),
            interactive=False,
        )

    predict_button.click(
        fn=predict_axial_frequency,
        inputs=[
            e_fixed_input,
            rho_fixed_input,
            nu_fixed_input,
            e_free_input,
            rho_free_input,
            nu_free_input,
        ],
        outputs=[
            result_output,
            status_output,
            validity_output,
        ],
    )


# =========================================================
# START APPLICATION
# =========================================================
if __name__ == "__main__":
    demo.launch()
