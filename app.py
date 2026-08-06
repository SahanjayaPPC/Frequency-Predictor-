from pathlib import Path
import html
import math

import gradio as gr
import joblib
import matplotlib

# Required for Hugging Face/headless environments
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

TOP_IMAGE_PATH = BASE_DIR / "Steel and Aluminium.png"
VIDEO_PATH = BASE_DIR / "Axial Deformation.mp4"

OUTPUT_DIR = BASE_DIR / "axial_frequency_outputs_xlsx"


# =========================================================
# MODEL INFORMATION
# Value obtained from the completed 5-fold CV analysis
# =========================================================
PREDICTOR_R2 = 0.9970


# =========================================================
# MODEL INPUT FEATURES
# These names must exactly match the training columns.
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
# BASIC HELPERS
# =========================================================
def first_existing_path(*candidates):
    """Return the first existing path, otherwise None."""
    for candidate in candidates:
        path = Path(candidate)

        if path.exists():
            return path

    return None


def format_scientific(value: float) -> str:
    """Format a number using scientific notation."""
    return f"{float(value):.6e}"


def parse_scientific_value(
    value,
    field_name: str,
) -> float:
    """
    Convert text into a floating-point value.

    Accepted formats:
        1.970000e+11
        1.97E11
        197000000000
        7,750.3
    """
    if value is None:
        raise ValueError(
            f"{field_name} is required."
        )

    text = str(value).strip().replace(",", "")

    if not text:
        raise ValueError(
            f"{field_name} is required."
        )

    try:
        number = float(text)

    except ValueError as error:
        raise ValueError(
            f"{field_name} must be a valid number. "
            "Scientific notation such as 1.97e11 is accepted."
        ) from error

    if not math.isfinite(number):
        raise ValueError(
            f"{field_name} must be a finite number."
        )

    return number


# =========================================================
# COLUMN NORMALIZATION
# =========================================================
def simplify_column_name(name: str) -> str:
    """
    Convert different Excel column formats into a
    comparable simplified format.

    Examples:
        ρ Fixed -> rhofixed
        ν Free  -> nufree
        E_Free  -> efree
    """
    text = str(name).strip().lower()

    text = text.replace("ρ", "rho")
    text = text.replace("ν", "nu")

    characters_to_remove = [
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
    ]

    for character in characters_to_remove:
        text = text.replace(character, "")

    return text


def normalize_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Rename dataset columns into canonical names."""
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
        simplified = simplify_column_name(
            original_column
        )

        if simplified in canonical_names:
            rename_dictionary[original_column] = (
                canonical_names[simplified]
            )

    return dataframe.rename(
        columns=rename_dictionary
    )


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

    reference_dataframe = pd.read_excel(
        DATA_PATH
    )

    reference_dataframe = normalize_columns(
        reference_dataframe
    )

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

    if REFERENCE_DATA.empty:
        raise ValueError(
            "No valid material-property rows were found "
            "in Raw_Data_Set.xlsx."
        )

except Exception as error:
    DATA_ERROR = str(error)


# =========================================================
# CALCULATE VARIABLE BOUNDARIES
# Boundaries are rebuilt automatically from the Excel file.
# =========================================================
def calculate_boundaries(
    dataframe: pd.DataFrame,
) -> dict:
    if dataframe is None or dataframe.empty:
        return {}

    boundaries = {}

    for feature in FEATURE_COLS:
        values = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        boundaries[feature] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "median": float(values.median()),
        }

    return boundaries


BOUNDARIES = calculate_boundaries(
    REFERENCE_DATA
)


def create_boundary_table() -> pd.DataFrame:
    rows = []

    for feature in FEATURE_COLS:
        boundary = BOUNDARIES.get(feature)

        if boundary is None:
            rows.append(
                {
                    "Variable": DISPLAY_NAMES[feature],
                    "Training Minimum": "N/A",
                    "Training Maximum": "N/A",
                }
            )

        else:
            rows.append(
                {
                    "Variable": DISPLAY_NAMES[feature],
                    "Training Minimum": format_scientific(
                        boundary["minimum"]
                    ),
                    "Training Maximum": format_scientific(
                        boundary["maximum"]
                    ),
                }
            )

    return pd.DataFrame(rows)


BOUNDARY_TABLE = create_boundary_table()


def range_information(feature: str) -> str:
    boundary = BOUNDARIES.get(feature)

    if boundary is None:
        return "Training range unavailable"

    return (
        "Training range: "
        f"{format_scientific(boundary['minimum'])} "
        "to "
        f"{format_scientific(boundary['maximum'])}"
    )


# =========================================================
# DEFAULT VALUES
# Preferred defaults are used only when they are inside
# the current dataset boundaries.
# =========================================================
def safe_default(
    feature: str,
    preferred_value: float,
) -> float:
    boundary = BOUNDARIES.get(feature)

    if boundary is None:
        return float(preferred_value)

    minimum = boundary["minimum"]
    maximum = boundary["maximum"]

    if minimum <= preferred_value <= maximum:
        return float(preferred_value)

    return float(boundary["median"])


DEFAULT_VALUES = {
    # Fixed material: steel
    "E Fixed": safe_default(
        "E Fixed",
        1.970000e11,
    ),
    "rho Fixed": safe_default(
        "rho Fixed",
        7.750300e03,
    ),
    "nu Fixed": safe_default(
        "nu Fixed",
        2.900000e-01,
    ),

    # Free material: aluminium
    "E Free": safe_default(
        "E Free",
        7.170000e10,
    ),
    "rho Free": safe_default(
        "rho Free",
        2.795700e03,
    ),
    "nu Free": safe_default(
        "nu Free",
        3.300000e-01,
    ),
}


# =========================================================
# CREATE MODEL INPUT DATAFRAME
# =========================================================
def create_input_dataframe(
    e_fixed: float,
    rho_fixed: float,
    nu_fixed: float,
    e_free: float,
    rho_free: float,
    nu_free: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "E Fixed": e_fixed,
                "rho Fixed": rho_fixed,
                "nu Fixed": nu_fixed,
                "E Free": e_free,
                "rho Free": rho_free,
                "nu Free": nu_free,
            }
        ]
    )


# =========================================================
# PHYSICAL INPUT VALIDATION
# =========================================================
def validate_physical_inputs(
    e_fixed: float,
    rho_fixed: float,
    nu_fixed: float,
    e_free: float,
    rho_free: float,
    nu_free: float,
) -> None:
    if e_fixed <= 0:
        raise ValueError(
            "E (Fixed) must be greater than zero."
        )

    if rho_fixed <= 0:
        raise ValueError(
            "ρ (Fixed) must be greater than zero."
        )

    if e_free <= 0:
        raise ValueError(
            "E (Free) must be greater than zero."
        )

    if rho_free <= 0:
        raise ValueError(
            "ρ (Free) must be greater than zero."
        )

    if not 0 <= nu_fixed < 0.5:
        raise ValueError(
            "ν (Fixed) must be between 0 and 0.5."
        )

    if not 0 <= nu_free < 0.5:
        raise ValueError(
            "ν (Free) must be between 0 and 0.5."
        )


# =========================================================
# TRAINING VALIDITY CHECK
# =========================================================
def build_validity_table(
    input_dataframe: pd.DataFrame,
):
    rows = []
    outside_messages = []

    for feature in FEATURE_COLS:
        current_value = float(
            input_dataframe.iloc[0][feature]
        )

        boundary = BOUNDARIES.get(feature)

        if boundary is None:
            minimum_display = "N/A"
            maximum_display = "N/A"
            status = "Not checked"

        else:
            minimum_value = boundary["minimum"]
            maximum_value = boundary["maximum"]

            minimum_display = format_scientific(
                minimum_value
            )

            maximum_display = format_scientific(
                maximum_value
            )

            if current_value < minimum_value:
                status = "Below range"

                outside_messages.append(
                    f"**{DISPLAY_NAMES[feature]}** is below "
                    "the training minimum: "
                    f"`{format_scientific(current_value)}` < "
                    f"`{format_scientific(minimum_value)}`."
                )

            elif current_value > maximum_value:
                status = "Above range"

                outside_messages.append(
                    f"**{DISPLAY_NAMES[feature]}** is above "
                    "the training maximum: "
                    f"`{format_scientific(current_value)}` > "
                    f"`{format_scientific(maximum_value)}`."
                )

            else:
                status = "Inside"

        rows.append(
            {
                "Feature": DISPLAY_NAMES[feature],
                "Current Value": format_scientific(
                    current_value
                ),
                "Training Minimum": minimum_display,
                "Training Maximum": maximum_display,
                "Status": status,
            }
        )

    return pd.DataFrame(rows), outside_messages


# =========================================================
# CONTOUR MAP CASES
# =========================================================
def build_single_material_cases(
    input_dataframe: pd.DataFrame,
):
    """
    Create:
        1. Fixed material used at both positions
        2. Free material used at both positions
    """
    row = input_dataframe.iloc[0]

    fixed_fixed_case = pd.DataFrame(
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

    free_free_case = pd.DataFrame(
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

    return fixed_fixed_case, free_free_case


# =========================================================
# BUILD CONTOUR BACKGROUND
# =========================================================
def build_contour_dataframe(
    reference_dataframe: pd.DataFrame,
    model,
) -> pd.DataFrame:
    if reference_dataframe is None:
        raise ValueError(
            "Reference dataset is unavailable."
        )

    mixed_cases = (
        reference_dataframe[FEATURE_COLS]
        .dropna()
        .copy()
    )

    if mixed_cases.empty:
        raise ValueError(
            "No complete rows are available for "
            "the contour map."
        )

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
    fixed_fixed_case, free_free_case = (
        build_single_material_cases(
            input_dataframe
        )
    )

    free_free_prediction = float(
        MODEL.predict(
            free_free_case[FEATURE_COLS]
        )[0]
    )

    fixed_fixed_prediction = float(
        MODEL.predict(
            fixed_fixed_case[FEATURE_COLS]
        )[0]
    )

    mixed_prediction = float(
        MODEL.predict(
            input_dataframe[FEATURE_COLS]
        )[0]
    )

    return (
        free_free_prediction,
        fixed_fixed_prediction,
        mixed_prediction,
    )


# =========================================================
# CREATE CONTOUR FIGURE
# =========================================================
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
            "Not enough unique contour points are available."
        )

    figure, axis = plt.subplots(
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

        contour = axis.tricontourf(
            triangulation,
            z_values,
            levels=20,
        )

    except Exception:
        contour = axis.scatter(
            x_values,
            y_values,
            c=z_values,
            s=45,
        )

    colorbar = figure.colorbar(
        contour,
        ax=axis,
    )

    colorbar.set_label(
        "Predicted axial frequency (Hz)"
    )

    x_minimum = min(
        float(np.min(x_values)),
        x_user,
    )

    x_maximum = max(
        float(np.max(x_values)),
        x_user,
    )

    y_minimum = min(
        float(np.min(y_values)),
        y_user,
    )

    y_maximum = max(
        float(np.max(y_values)),
        y_user,
    )

    x_padding = 0.05 * (
        x_maximum - x_minimum
        if x_maximum > x_minimum
        else 1.0
    )

    y_padding = 0.05 * (
        y_maximum - y_minimum
        if y_maximum > y_minimum
        else 1.0
    )

    x_lower = x_minimum - x_padding
    x_upper = x_maximum + x_padding

    y_lower = y_minimum - y_padding
    y_upper = y_maximum + y_padding

    axis.set_xlim(
        x_lower,
        x_upper,
    )

    axis.set_ylim(
        y_lower,
        y_upper,
    )

    axis.axvline(
        x_user,
        linestyle="--",
        linewidth=1.8,
        label="Free material – Free material",
    )

    axis.axhline(
        y_user,
        linestyle="--",
        linewidth=1.8,
        label="Fixed material – Fixed material",
    )

    axis.scatter(
        [x_user],
        [y_lower],
        s=90,
        marker="s",
        zorder=6,
    )

    axis.annotate(
        f"Free – Free\n{x_user:.2f} Hz",
        (x_user, y_lower),
        textcoords="offset points",
        xytext=(15, 20),
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.82,
        },
    )

    axis.scatter(
        [x_lower],
        [y_user],
        s=90,
        marker="^",
        zorder=6,
    )

    axis.annotate(
        f"Fixed – Fixed\n{y_user:.2f} Hz",
        (x_lower, y_user),
        textcoords="offset points",
        xytext=(18, 18),
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.82,
        },
    )

    axis.scatter(
        [x_user],
        [y_user],
        s=170,
        facecolors="none",
        edgecolors="white",
        linewidths=2.7,
        zorder=7,
        label="Mixed-material intersection",
    )

    axis.scatter(
        [x_user],
        [y_user],
        s=35,
        zorder=8,
    )

    axis.annotate(
        f"Mixed material\n{z_user:.2f} Hz",
        (x_user, y_user),
        textcoords="offset points",
        xytext=(32, 42),
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.82,
        },
    )

    axis.set_xlabel(
        "Free material – Free material predicted "
        "axial frequency (Hz)"
    )

    axis.set_ylabel(
        "Fixed material – Fixed material predicted "
        "axial frequency (Hz)"
    )

    axis.set_title(
        "Contour Map of Predicted Axial Frequency"
    )

    axis.grid(
        True,
        alpha=0.30,
    )

    axis.legend(
        loc="best"
    )

    figure.tight_layout()

    return figure


# =========================================================
# MODEL OUTPUT IMAGE GALLERY
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
        "GAM Partial-Effect Plots",
    ),
]


def collect_output_images():
    images = []

    for filename, caption in OUTPUT_IMAGE_DEFINITIONS:
        image_path = first_existing_path(
            BASE_DIR / filename,
            OUTPUT_DIR / filename,
        )

        if image_path is not None:
            images.append(
                (
                    str(image_path),
                    caption,
                )
            )

    return images


OUTPUT_IMAGES = collect_output_images()


# =========================================================
# HTML RESULT CARDS
# =========================================================
def build_prediction_card(
    prediction: float,
) -> str:
    return f"""
    <div class="prediction-card">
        <div class="prediction-title">
            ExtraTrees Predictor
        </div>

        <div class="prediction-number">
            {prediction:,.2f}
            <span class="prediction-unit">
                Hz
            </span>
        </div>
    </div>
    """


def build_r2_card() -> str:
    return f"""
    <div class="metric-card-row">

        <div class="metric-card">
            <div class="metric-title">
                Predictor R²
            </div>

            <div class="metric-value">
                {PREDICTOR_R2:.4f}
            </div>

            <div class="metric-caption">
                Mean R² from 5-fold cross-validation
            </div>
        </div>

    </div>
    """


def build_contour_value_cards(
    free_free: float,
    fixed_fixed: float,
    mixed: float,
) -> str:
    return f"""
    <div class="three-card-row">

        <div class="information-card">
            <div class="information-title">
                Free Material – Free Material
            </div>

            <div class="information-value">
                {free_free:,.2f} Hz
            </div>
        </div>

        <div class="information-card">
            <div class="information-title">
                Fixed Material – Fixed Material
            </div>

            <div class="information-value">
                {fixed_fixed:,.2f} Hz
            </div>
        </div>

        <div class="information-card">
            <div class="information-title">
                Mixed-Material Frequency
            </div>

            <div class="information-value">
                {mixed:,.2f} Hz
            </div>
        </div>

    </div>
    """


# =========================================================
# MAIN PREDICTION FUNCTION
#
# ZeroGPU requires at least one @spaces.GPU function.
# The ExtraTrees prediction itself is CPU-based.
# =========================================================
@spaces.GPU(duration=10)
def predict_axial_frequency(
    e_fixed_text,
    rho_fixed_text,
    nu_fixed_text,
    e_free_text,
    rho_free_text,
    nu_free_text,
):
    try:
        if MODEL is None:
            raise RuntimeError(
                "ExtraTrees.pkl could not be loaded. "
                f"Details: {MODEL_ERROR}"
            )

        e_fixed = parse_scientific_value(
            e_fixed_text,
            "E (Fixed)",
        )

        rho_fixed = parse_scientific_value(
            rho_fixed_text,
            "ρ (Fixed)",
        )

        nu_fixed = parse_scientific_value(
            nu_fixed_text,
            "ν (Fixed)",
        )

        e_free = parse_scientific_value(
            e_free_text,
            "E (Free)",
        )

        rho_free = parse_scientific_value(
            rho_free_text,
            "ρ (Free)",
        )

        nu_free = parse_scientific_value(
            nu_free_text,
            "ν (Free)",
        )

        validate_physical_inputs(
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
            build_validity_table(
                input_dataframe
            )
        )

        if outside_messages:
            message_list = "\n".join(
                f"- {message}"
                for message in outside_messages
            )

            status_message = (
                "⚠️ **Input values outside the training "
                "region:**\n\n"
                f"{message_list}\n\n"
                "The model can still produce a result, but "
                "the prediction may be less reliable."
            )

        elif REFERENCE_DATA is None:
            status_message = (
                "ℹ️ Prediction completed, but the training "
                "ranges could not be checked. "
                f"Dataset error: {DATA_ERROR}"
            )

        else:
            status_message = (
                "✅ All six values are inside the "
                "current dataset training ranges."
            )

        contour_figure = None
        contour_cards = ""

        if CONTOUR_DATA is not None:
            (
                free_free_prediction,
                fixed_fixed_prediction,
                mixed_prediction,
            ) = create_user_contour_coordinates(
                input_dataframe
            )

            contour_figure = create_contour_plot(
                CONTOUR_DATA,
                free_free_prediction,
                fixed_fixed_prediction,
                mixed_prediction,
            )

            contour_cards = build_contour_value_cards(
                free_free_prediction,
                fixed_fixed_prediction,
                mixed_prediction,
            )

        elif CONTOUR_ERROR:
            status_message += (
                "\n\nContour-map error: "
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
            "Please correct the input values or inspect "
            "the Space logs.",
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

.scientific-input input {
    font-family: "Courier New", monospace !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.02rem !important;
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

.metric-card-row {
    display: grid;
    grid-template-columns: minmax(280px, 460px);
    gap: 20px;
    margin-top: 12px;
    margin-bottom: 22px;
}

.metric-card {
    background: #0d1d2a;
    border: 1px solid #1e6a9e;
    border-radius: 18px;
    padding: 24px 28px;
    min-height: 125px;
}

.metric-title {
    color: #ffffff;
    font-size: 1.12rem;
    font-weight: 750;
    margin-bottom: 14px;
}

.metric-value {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 800;
}

.metric-caption {
    color: #9ca3af;
    font-size: 0.88rem;
    margin-top: 8px;
}

.three-card-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 12px;
    margin-bottom: 18px;
}

.information-card {
    background: rgba(33, 150, 243, 0.13);
    border: 1px solid rgba(33, 150, 243, 0.35);
    border-radius: 14px;
    padding: 18px;
}

.information-title {
    color: #9fd0ff;
    font-size: 0.95rem;
    font-weight: 650;
    margin-bottom: 10px;
}

.information-value {
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
    .metric-card-row {
        grid-template-columns: 1fr;
    }

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
        Enter the material properties using scientific notation,
        for example <strong>1.970000e+11</strong>.
        The training ranges are calculated automatically from
        Raw_Data_Set.xlsx.
        </div>
        """
    )

    if TOP_IMAGE_PATH.exists():
        gr.Image(
            value=str(TOP_IMAGE_PATH),
            show_label=False,
            interactive=False,
        )

    with gr.Accordion(
        "Current Dataset Boundaries",
        open=False,
    ):
        gr.Dataframe(
            value=BOUNDARY_TABLE,
            interactive=False,
        )

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Fixed Material")

            e_fixed_input = gr.Textbox(
                label="E (Fixed) [N/m²]",
                value=format_scientific(
                    DEFAULT_VALUES["E Fixed"]
                ),
                placeholder="Example: 1.970000e+11",
                info=range_information("E Fixed"),
                elem_classes=["scientific-input"],
            )

            rho_fixed_input = gr.Textbox(
                label="ρ (Fixed) [kg/m³]",
                value=format_scientific(
                    DEFAULT_VALUES["rho Fixed"]
                ),
                placeholder="Example: 7.750300e+03",
                info=range_information("rho Fixed"),
                elem_classes=["scientific-input"],
            )

            nu_fixed_input = gr.Textbox(
                label="ν (Fixed) [-]",
                value=format_scientific(
                    DEFAULT_VALUES["nu Fixed"]
                ),
                placeholder="Example: 2.900000e-01",
                info=range_information("nu Fixed"),
                elem_classes=["scientific-input"],
            )

        with gr.Column():
            gr.Markdown("## Free Material")

            e_free_input = gr.Textbox(
                label="E (Free) [N/m²]",
                value=format_scientific(
                    DEFAULT_VALUES["E Free"]
                ),
                placeholder="Example: 7.170000e+10",
                info=range_information("E Free"),
                elem_classes=["scientific-input"],
            )

            rho_free_input = gr.Textbox(
                label="ρ (Free) [kg/m³]",
                value=format_scientific(
                    DEFAULT_VALUES["rho Free"]
                ),
                placeholder="Example: 2.795700e+03",
                info=range_information("rho Free"),
                elem_classes=["scientific-input"],
            )

            nu_free_input = gr.Textbox(
                label="ν (Free) [-]",
                value=format_scientific(
                    DEFAULT_VALUES["nu Free"]
                ),
                placeholder="Example: 3.300000e-01",
                info=range_information("nu Free"),
                elem_classes=["scientific-input"],
            )

    predict_button = gr.Button(
        "Predict Axial Frequency",
        variant="primary",
    )

    result_output = gr.HTML()

    # Fixed model-performance value from completed analysis
    gr.HTML(
        value=build_r2_card()
    )

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
        - **intersection:** the current mixed-material
          ExtraTrees prediction.
        """
    )

    contour_output = gr.Plot(
        label="ExtraTrees Frequency Contour",
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
                height=1000,
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
# START APPLICATION
# =========================================================
if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1
    )

    demo.launch(
        css=CUSTOM_CSS,
    )
