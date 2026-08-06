from pathlib import Path
import html

import gradio as gr
import joblib
import matplotlib

# Required for plotting on Hugging Face without a desktop display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import spaces


# =========================================================
# FILE PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ExtraTrees.pkl"
DATA_PATH = BASE_DIR / "Raw_Data_Set.xlsx"

IMAGE_PATH = BASE_DIR / "Steel and Aluminium.png"
VIDEO_PATH = BASE_DIR / "Axial Deformation.mp4"

OUTPUT_DIR = BASE_DIR / "axial_frequency_outputs_xlsx"


# =========================================================
# MODEL FEATURES
# These names must match the model training columns exactly.
# =========================================================
FEATURE_COLS = [
    "E Fixed",
    "rho Fixed",
    "nu Fixed",
    "E Free",
    "rho Free",
    "nu Free",
]

TARGET_COL = "Axial Frequency (Hz)"


DISPLAY_NAMES = {
    "E Fixed": "E (Fixed) [N/m²]",
    "rho Fixed": "ρ (Fixed) [kg/m³]",
    "nu Fixed": "ν (Fixed) [-]",
    "E Free": "E (Free) [N/m²]",
    "rho Free": "ρ (Free) [kg/m³]",
    "nu Free": "ν (Free) [-]",
}


# =========================================================
# FILE HELPERS
# =========================================================
def first_existing_path(*candidates):
    """Return the first existing file from the candidates."""
    for candidate in candidates:
        path = Path(candidate)

        if path.exists():
            return path

    return None


# =========================================================
# COLUMN NORMALIZATION
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
# LOAD EXTRATREES MODEL
# =========================================================
MODEL = None
MODEL_ERROR = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH.name}"
        )

    MODEL = joblib.load(MODEL_PATH)

except Exception as error:
    MODEL_ERROR = str(error)


# =========================================================
# LOAD REFERENCE DATASET
# =========================================================
REFERENCE_DATA = None
DATA_ERROR = None

try:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {DATA_PATH.name}"
        )

    reference_dataframe = pd.read_excel(DATA_PATH)
    reference_dataframe = normalize_columns(reference_dataframe)

    missing_columns = [
        column
        for column in FEATURE_COLS
        if column not in reference_dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following columns are missing from "
            f"Raw_Data_Set.xlsx: {missing_columns}"
        )

    columns_to_keep = FEATURE_COLS.copy()

    if TARGET_COL in reference_dataframe.columns:
        columns_to_keep.append(TARGET_COL)

    reference_dataframe = reference_dataframe[
        columns_to_keep
    ].copy()

    for column in columns_to_keep:
        reference_dataframe[column] = pd.to_numeric(
            reference_dataframe[column],
            errors="coerce",
        )

    REFERENCE_DATA = (
        reference_dataframe
        .dropna(subset=FEATURE_COLS, how="all")
        .reset_index(drop=True)
    )

except Exception as error:
    DATA_ERROR = str(error)


# =========================================================
# CREATE INPUT DATAFRAME
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
# TRAINING VALIDITY TABLE
# =========================================================
def build_validity_table(
    input_dataframe: pd.DataFrame,
):
    rows = []
    outside_messages = []

    if REFERENCE_DATA is None or REFERENCE_DATA.empty:
        for feature in FEATURE_COLS:
            current_value = float(
                input_dataframe.iloc[0][feature]
            )

            rows.append(
                {
                    "Feature": DISPLAY_NAMES[feature],
                    "Current Value": f"{current_value:.6e}",
                    "Training Minimum": "N/A",
                    "Training Maximum": "N/A",
                    "Status": "Not checked",
                }
            )

        return pd.DataFrame(rows), outside_messages

    for feature in FEATURE_COLS:
        current_value = float(
            input_dataframe.iloc[0][feature]
        )

        reference_values = (
            REFERENCE_DATA[feature]
            .dropna()
        )

        minimum_value = float(reference_values.min())
        maximum_value = float(reference_values.max())

        if current_value < minimum_value:
            status = "Below range"

            outside_messages.append(
                f"**{DISPLAY_NAMES[feature]}** is below the "
                f"training minimum: `{current_value:.6e}` "
                f"< `{minimum_value:.6e}`."
            )

        elif current_value > maximum_value:
            status = "Above range"

            outside_messages.append(
                f"**{DISPLAY_NAMES[feature]}** is above the "
                f"training maximum: `{current_value:.6e}` "
                f"> `{maximum_value:.6e}`."
            )

        else:
            status = "Inside"

        rows.append(
            {
                "Feature": DISPLAY_NAMES[feature],
                "Current Value": f"{current_value:.6e}",
                "Training Minimum": f"{minimum_value:.6e}",
                "Training Maximum": f"{maximum_value:.6e}",
                "Status": status,
            }
        )

    return pd.DataFrame(rows), outside_messages


# =========================================================
# CONTOUR MAP DATA
# =========================================================
def build_single_material_cases(
    input_dataframe: pd.DataFrame,
):
    """
    Create:

    1. Fixed material used at both ends
    2. Free material used at both ends
    """
    row = input_dataframe.iloc[0]

    fixed_fixed = pd.DataFrame(
        [
            {
                "E Fixed": row["E Fixed"],
                "rho Fixed": row["rho Fixed"],
                "nu Fixed": row["nu Fixed"],
                "E Free": row["E Fixed"],
                "rho Free": row["rho Fixed"],
                "nu Free": row["nu Fixed"],
            }
        ]
    )

    free_free = pd.DataFrame(
        [
            {
                "E Fixed": row["E Free"],
                "rho Fixed": row["rho Free"],
                "nu Fixed": row["nu Free"],
                "E Free": row["E Free"],
                "rho Free": row["rho Free"],
                "nu Free": row["nu Free"],
            }
        ]
    )

    return fixed_fixed, free_free


def build_contour_dataframe(
    reference_dataframe: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Create the background contour-map dataset.

    x-axis:
        Free material used at both ends

    y-axis:
        Fixed material used at both ends

    z-value:
        Mixed-material prediction
    """
    mixed_cases = reference_dataframe[
        FEATURE_COLS
    ].copy()

    fixed_fixed_cases = pd.DataFrame(
        {
            "E Fixed": mixed_cases["E Fixed"],
            "rho Fixed": mixed_cases["rho Fixed"],
            "nu Fixed": mixed_cases["nu Fixed"],
            "E Free": mixed_cases["E Fixed"],
            "rho Free": mixed_cases["rho Fixed"],
            "nu Free": mixed_cases["nu Fixed"],
        }
    )

    free_free_cases = pd.DataFrame(
        {
            "E Fixed": mixed_cases["E Free"],
            "rho Fixed": mixed_cases["rho Free"],
            "nu Fixed": mixed_cases["nu Free"],
            "E Free": mixed_cases["E Free"],
            "rho Free": mixed_cases["rho Free"],
            "nu Free": mixed_cases["nu Free"],
        }
    )

    x_axis = model.predict(
        free_free_cases[FEATURE_COLS]
    )

    y_axis = model.predict(
        fixed_fixed_cases[FEATURE_COLS]
    )

    z_values = model.predict(
        mixed_cases[FEATURE_COLS]
    )

    contour_dataframe = pd.DataFrame(
        {
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_value": z_values,
        }
    )

    contour_dataframe = (
        contour_dataframe
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .drop_duplicates(
            subset=["x_axis", "y_axis"]
        )
        .reset_index(drop=True)
    )

    return contour_dataframe


CONTOUR_DATA = None
CONTOUR_ERROR = None

try:
    if (
        MODEL is not None
        and REFERENCE_DATA is not None
        and not REFERENCE_DATA.empty
    ):
        CONTOUR_DATA = build_contour_dataframe(
            REFERENCE_DATA,
            MODEL,
        )

except Exception as error:
    CONTOUR_ERROR = str(error)


def create_user_contour_coordinates(
    input_dataframe: pd.DataFrame,
):
    fixed_fixed, free_free = (
        build_single_material_cases(input_dataframe)
    )

    x_axis = float(
        MODEL.predict(
            free_free[FEATURE_COLS]
        )[0]
    )

    y_axis = float(
        MODEL.predict(
            fixed_fixed[FEATURE_COLS]
        )[0]
    )

    mixed_prediction = float(
        MODEL.predict(
            input_dataframe[FEATURE_COLS]
        )[0]
    )

    return x_axis, y_axis, mixed_prediction


def create_contour_plot(
    contour_dataframe: pd.DataFrame,
    x_user: float,
    y_user: float,
    z_user: float,
):
    if (
        contour_dataframe is None
        or len(contour_dataframe) < 3
    ):
        raise ValueError(
            "Not enough contour-map points are available."
        )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    x_values = contour_dataframe[
        "x_axis"
    ].to_numpy()

    y_values = contour_dataframe[
        "y_axis"
    ].to_numpy()

    z_values = contour_dataframe[
        "z_value"
    ].to_numpy()

    try:
        triangulation = mtri.Triangulation(
            x_values,
            y_values,
        )

        contour = ax.tricontourf(
            triangulation,
            z_values,
            levels=20,
        )

    except Exception:
        # Fallback if the points cannot be triangulated
        contour = ax.scatter(
            x_values,
            y_values,
            c=z_values,
            s=45,
        )

    colorbar = fig.colorbar(
        contour,
        ax=ax,
    )

    colorbar.set_label(
        r"Predicted $f_{\mathrm{axial}}$ (Hz)"
    )

    x_min = min(
        float(np.min(x_values)),
        x_user,
    )

    x_max = max(
        float(np.max(x_values)),
        x_user,
    )

    y_min = min(
        float(np.min(y_values)),
        y_user,
    )

    y_max = max(
        float(np.max(y_values)),
        y_user,
    )

    x_padding = 0.05 * (
        x_max - x_min
        if x_max > x_min
        else 1.0
    )

    y_padding = 0.05 * (
        y_max - y_min
        if y_max > y_min
        else 1.0
    )

    x_lower = x_min - x_padding
    x_upper = x_max + x_padding

    y_lower = y_min - y_padding
    y_upper = y_max + y_padding

    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)

    # Free/free prediction line
    ax.axvline(
        x_user,
        linestyle="--",
        linewidth=1.8,
        label="Free material – Free material",
    )

    # Fixed/fixed prediction line
    ax.axhline(
        y_user,
        linestyle="--",
        linewidth=1.8,
        label="Fixed material – Fixed material",
    )

    # Free/free marker
    ax.scatter(
        [x_user],
        [y_lower],
        s=90,
        marker="s",
        zorder=6,
    )

    ax.annotate(
        f"Free – Free\n{x_user:.2f} Hz",
        (x_user, y_lower),
        textcoords="offset points",
        xytext=(15, 20),
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.80,
        },
    )

    # Fixed/fixed marker
    ax.scatter(
        [x_lower],
        [y_user],
        s=90,
        marker="^",
        zorder=6,
    )

    ax.annotate(
        f"Fixed – Fixed\n{y_user:.2f} Hz",
        (x_lower, y_user),
        textcoords="offset points",
        xytext=(18, 18),
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.80,
        },
    )

    # Mixed-material intersection
    ax.scatter(
        [x_user],
        [y_user],
        s=170,
        facecolors="none",
        edgecolors="white",
        linewidths=2.7,
        zorder=7,
        label="Mixed-material intersection",
    )

    ax.scatter(
        [x_user],
        [y_user],
        s=35,
        zorder=8,
    )

    ax.annotate(
        f"Mixed material\n{z_user:.2f} Hz",
        (x_user, y_user),
        textcoords="offset points",
        xytext=(32, 42),
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.80,
        },
    )

    ax.set_xlabel(
        "Free material – Free material "
        "predicted axial frequency (Hz)"
    )

    ax.set_ylabel(
        "Fixed material – Fixed material "
        "predicted axial frequency (Hz)"
    )

    ax.set_title(
        "Contour Map of Predicted Axial Frequency"
    )

    ax.grid(
        True,
        alpha=0.30,
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    return fig


# =========================================================
# OUTPUT IMAGE GALLERY
# =========================================================
OUTPUT_IMAGE_DEFINITIONS = [
    (
        "correlation_heatmap.png",
        "Correlation Heatmap",
    ),
    (
        "cv_rmse_by_fold.png",
        "Cross-Validation RMSE",
    ),
    (
        "cv_r2_by_fold.png",
        "Cross-Validation R²",
    ),
    (
        "parity_plot_ExtraTrees.png",
        "ExtraTrees Parity Plot",
    ),
    (
        "permutation_importance_ExtraTrees.png",
        "ExtraTrees Permutation Importance",
    ),
    (
        "gam_partial_effect_plots.png",
        "GAM Partial Effects",
    ),
]


def collect_output_images():
    gallery_images = []

    for filename, caption in OUTPUT_IMAGE_DEFINITIONS:
        image_path = first_existing_path(
            BASE_DIR / filename,
            OUTPUT_DIR / filename,
        )

        if image_path is not None:
            gallery_images.append(
                (
                    str(image_path),
                    caption,
                )
            )

    return gallery_images


OUTPUT_IMAGES = collect_output_images()


# =========================================================
# RESULT CARDS
# =========================================================
def build_prediction_card(
    prediction: float,
) -> str:
    return f"""
    <div class="prediction-card">
        <div class="prediction-title">
            Predicted Axial Frequency
        </div>

        <div class="prediction-number">
            {prediction:,.2f}
            <span class="prediction-unit">
                Hz
            </span>
        </div>

        <div class="prediction-model">
            Model: ExtraTrees Regressor
        </div>
    </div>
    """


def build_contour_value_cards(
    x_user: float,
    y_user: float,
    z_user: float,
) -> str:
    return f"""
    <div class="three-card-row">

        <div class="info-card">
            <div class="info-title">
                Free Material – Free Material
            </div>

            <div class="info-value">
                {x_user:,.2f} Hz
            </div>
        </div>

        <div class="info-card">
            <div class="info-title">
                Fixed Material – Fixed Material
            </div>

            <div class="info-value">
                {y_user:,.2f} Hz
            </div>
        </div>

        <div class="info-card">
            <div class="info-title">
                Mixed-Material Frequency
            </div>

            <div class="info-value">
                {z_user:,.2f} Hz
            </div>
        </div>

    </div>
    """


# =========================================================
# PREDICTION FUNCTION
# ZeroGPU requires a @spaces.GPU decorated function.
# =========================================================
@spaces.GPU(duration=10)
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
                "ExtraTrees.pkl could not be loaded. "
                f"Details: {MODEL_ERROR}"
            )

        validate_inputs(
            e_fixed,
            rho_fixed,
            nu_fixed,
            e_free,
            rho_free,
            nu_free,
        )

        input_dataframe = create_input_dataframe(
            e_fixed,
            rho_fixed,
            nu_fixed,
            e_free,
            rho_free,
            nu_free,
        )

        prediction = float(
            MODEL.predict(
                input_dataframe[FEATURE_COLS]
            )[0]
        )

        validity_table, outside_messages = (
            build_validity_table(input_dataframe)
        )

        if outside_messages:
            bullet_list = "\n".join(
                f"- {message}"
                for message in outside_messages
            )

            status_message = (
                "⚠️ **Some values are outside the "
                "model-training range:**\n\n"
                f"{bullet_list}\n\n"
                "The model can still return a result, but "
                "the prediction may be less reliable."
            )

        elif REFERENCE_DATA is None:
            status_message = (
                "ℹ️ Prediction completed, but the training "
                "range could not be checked. "
                f"Dataset details: {DATA_ERROR}"
            )

        else:
            status_message = (
                "✅ All six input values are inside their "
                "training-data ranges."
            )

        x_user = None
        y_user = None
        z_user = prediction
        contour_figure = None
        contour_cards = ""

        if CONTOUR_DATA is not None:
            x_user, y_user, z_user = (
                create_user_contour_coordinates(
                    input_dataframe
                )
            )

            contour_figure = create_contour_plot(
                CONTOUR_DATA,
                x_user,
                y_user,
                z_user,
            )

            contour_cards = build_contour_value_cards(
                x_user,
                y_user,
                z_user,
            )

        elif CONTOUR_ERROR:
            status_message += (
                "\n\nContour map could not be generated: "
                f"`{CONTOUR_ERROR}`"
            )

        video_value = (
            str(VIDEO_PATH)
            if VIDEO_PATH.exists()
            else None
        )

        return (
            build_prediction_card(prediction),
            status_message,
            validity_table,
            contour_figure,
            contour_cards,
            video_value,
        )

    except Exception as error:
        escaped_error = html.escape(
            str(error)
        )

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
            "Please correct the inputs or inspect the Space logs.",
            empty_table,
            None,
            "",
            None,
        )


# =========================================================
# CUSTOM CSS
# =========================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 1350px !important;
    margin: auto !important;
}

.app-subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    margin-bottom: 1.3rem;
}

.prediction-card {
    background: linear-gradient(
        135deg,
        #0f5132,
        #198754
    );
    border-radius: 18px;
    padding: 30px;
    margin-top: 15px;
    margin-bottom: 18px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.24);
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

.three-card-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 12px;
    margin-bottom: 18px;
}

.info-card {
    background: rgba(33, 150, 243, 0.13);
    border: 1px solid rgba(33, 150, 243, 0.35);
    border-radius: 14px;
    padding: 18px;
}

.info-title {
    color: #9fd0ff;
    font-size: 0.95rem;
    font-weight: 650;
    margin-bottom: 10px;
}

.info-value {
    color: white;
    font-size: 1.25rem;
    font-weight: 750;
}

.error-card {
    background: rgba(220, 38, 38, 0.12);
    border: 1px solid rgba(220, 38, 38, 0.45);
    border-radius: 14px;
    padding: 20px;
    margin-top: 15px;
}

.error-title {
    color: #ef4444;
    font-size: 1.2rem;
    font-weight: 750;
    margin-bottom: 10px;
}

.error-message {
    font-size: 1rem;
}

@media (max-width: 850px) {
    .three-card-row {
        grid-template-columns: 1fr;
    }
}
"""


# =========================================================
# GRADIO GUI
# =========================================================
with gr.Blocks(
    title="Axial Frequency Predictor",
) as demo:

    gr.Markdown(
        """
        # ⚙️ Axial Frequency Predictor

        <div class="app-subtitle">
        Predict axial frequency using the trained
        ExtraTrees regression model.
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

    gr.Markdown(
        """
        ## Contour Map

        - **x-axis:** predicted frequency when the free material
          is used for both material positions.
        - **y-axis:** predicted frequency when the fixed material
          is used for both material positions.
        - **intersection:** current mixed-material ExtraTrees
          prediction.
        """
    )

    contour_output = gr.Plot(
        label="ExtraTrees Frequency Contour",
        format="png",
    )

    contour_values_output = gr.HTML()

    if VIDEO_PATH.exists():
        gr.Markdown("## Axial Deformation")

        video_output = gr.Video(
            value=None,
            label="Axial Deformation",
            interactive=False,
            autoplay=True,
            loop=True,
        )

    else:
        video_output = gr.Video(
            visible=False,
        )

    if OUTPUT_IMAGES:
        with gr.Accordion(
            "Model Validation and Analysis Images",
            open=False,
        ):
            gr.Gallery(
                value=OUTPUT_IMAGES,
                label="Model Output Images",
                columns=2,
                rows=3,
                height="1000px",
                object_fit="contain",
                allow_preview=True,
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
            contour_output,
            contour_values_output,
            video_output,
        ],
        scroll_to_output=True,
    )


# =========================================================
# START APP
# =========================================================
if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1
    )

    demo.launch(
        css=CUSTOM_CSS,
    )
