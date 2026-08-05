from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


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
# These names must exactly match the columns used in training.
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


# =========================================================
# STREAMLIT PAGE SETTINGS
# =========================================================
st.set_page_config(
    page_title="Axial Frequency Predictor",
    page_icon="⚙️",
    layout="wide",
)


# =========================================================
# CUSTOM STYLE
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 1.5rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #0f5132, #198754);
        border-radius: 16px;
        padding: 28px;
        margin-top: 12px;
        margin-bottom: 18px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.25);
    }

    .prediction-label {
        font-size: 1.15rem;
        font-weight: 650;
        color: #d9fbe5;
        margin-bottom: 12px;
    }

    .prediction-value {
        font-size: 2.8rem;
        font-weight: 850;
        color: #ffffff;
        line-height: 1.1;
    }

    .prediction-unit {
        font-size: 1.15rem;
        font-weight: 600;
        color: #d9fbe5;
        margin-left: 6px;
    }

    .info-box {
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(59, 130, 246, 0.30);
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 10px;
        margin-bottom: 14px;
    }

    .warning-box {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 14px 16px;
        color: #facc15;
        margin-top: 10px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# COLUMN NORMALIZATION
# Used only for the optional Excel validity table.
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    canonical_targets = {
        "efixed": "E Fixed",
        "rhofixed": "rho Fixed",
        "nufixed": "nu Fixed",
        "efree": "E Free",
        "rhofree": "rho Free",
        "nufree": "nu Free",
        "axialfrequencyhz": "Axial Frequency (Hz)",
        "axialfrequency": "Axial Frequency (Hz)",
    }

    rename_dict = {}

    for original_column in df.columns:
        simplified_column = simplify_name(original_column)

        if simplified_column in canonical_targets:
            rename_dict[original_column] = canonical_targets[
                simplified_column
            ]

    return df.rename(columns=rename_dict)


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH.name}"
        )

    return joblib.load(MODEL_PATH)


# =========================================================
# LOAD OPTIONAL DATASET
# =========================================================
@st.cache_data
def load_reference_dataset():
    if not DATA_PATH.exists():
        return None

    df = pd.read_excel(DATA_PATH)
    df = normalize_columns(df)

    missing_columns = [
        column
        for column in FEATURE_COLS
        if column not in df.columns
    ]

    if missing_columns:
        return None

    reference_df = df[FEATURE_COLS].copy()

    for column in FEATURE_COLS:
        reference_df[column] = pd.to_numeric(
            reference_df[column],
            errors="coerce",
        )

    return reference_df.dropna(how="all").reset_index(drop=True)


# =========================================================
# INPUT DATAFRAME
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
# VALIDITY-RANGE TABLE
# =========================================================
def build_validity_table(
    input_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    display_names = {
        "E Fixed": "E (Fixed) [N/m²]",
        "rho Fixed": "ρ (Fixed) [kg/m³]",
        "nu Fixed": "ν (Fixed) [-]",
        "E Free": "E (Free) [N/m²]",
        "rho Free": "ρ (Free) [kg/m³]",
        "nu Free": "ν (Free) [-]",
    }

    rows = []
    input_values = input_df.iloc[0]

    for feature in FEATURE_COLS:
        current_value = float(input_values[feature])
        minimum_value = float(reference_df[feature].min())
        maximum_value = float(reference_df[feature].max())

        status = (
            "Inside"
            if minimum_value <= current_value <= maximum_value
            else "Outside"
        )

        rows.append(
            {
                "Feature": display_names.get(feature, feature),
                "Current Value": f"{current_value:.6e}",
                "Training Minimum": f"{minimum_value:.6e}",
                "Training Maximum": f"{maximum_value:.6e}",
                "Status": status,
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# INPUT VALIDATION
# =========================================================
def validate_inputs(
    e_fixed: float,
    rho_fixed: float,
    nu_fixed: float,
    e_free: float,
    rho_free: float,
    nu_free: float,
):
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
# PAGE HEADER
# =========================================================
st.markdown(
    '<div class="main-title">Axial Frequency Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Predict axial frequency using a trained ExtraTrees regression model.
    Enter the material properties for the fixed and free sections.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# OPTIONAL IMAGE
# =========================================================
if IMAGE_PATH.exists():
    st.image(
        str(IMAGE_PATH),
        use_container_width=True,
    )


# =========================================================
# SIDEBAR INPUTS
# =========================================================
with st.sidebar:
    st.header("Input Material Properties")

    st.caption(
        "Use SI units: E in N/m², ρ in kg/m³, "
        "and ν is dimensionless."
    )

    st.subheader("Fixed Material")

    e_fixed = st.number_input(
        "E (Fixed) [N/m²]",
        min_value=0.0,
        value=1.97e11,
        step=1e8,
        format="%.6e",
    )

    rho_fixed = st.number_input(
        "ρ (Fixed) [kg/m³]",
        min_value=0.0,
        value=7750.3,
        step=1.0,
        format="%.6f",
    )

    nu_fixed = st.number_input(
        "ν (Fixed) [-]",
        min_value=0.0,
        max_value=0.4999,
        value=0.29,
        step=0.001,
        format="%.6f",
    )

    st.subheader("Free Material")

    e_free = st.number_input(
        "E (Free) [N/m²]",
        min_value=0.0,
        value=4.24e8,
        step=1e8,
        format="%.6e",
    )

    rho_free = st.number_input(
        "ρ (Free) [kg/m³]",
        min_value=0.0,
        value=2200.5,
        step=1.0,
        format="%.6f",
    )

    nu_free = st.number_input(
        "ν (Free) [-]",
        min_value=0.0,
        max_value=0.4999,
        value=0.45,
        step=0.001,
        format="%.6f",
    )

    predict_button = st.button(
        "Predict Axial Frequency",
        use_container_width=True,
        type="primary",
    )


# =========================================================
# PREDICTION
# =========================================================
if predict_button:
    try:
        validate_inputs(
            e_fixed=e_fixed,
            rho_fixed=rho_fixed,
            nu_fixed=nu_fixed,
            e_free=e_free,
            rho_free=rho_free,
            nu_free=nu_free,
        )

        model = load_model()

        input_df = create_input_dataframe(
            e_fixed=e_fixed,
            rho_fixed=rho_fixed,
            nu_fixed=nu_fixed,
            e_free=e_free,
            rho_free=rho_free,
            nu_free=nu_free,
        )

        prediction = float(
            model.predict(
                input_df[FEATURE_COLS]
            )[0]
        )

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">
                    Predicted Axial Frequency
                </div>

                <div class="prediction-value">
                    {prediction:,.2f}
                    <span class="prediction-unit">Hz</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Input Summary")

        summary_df = pd.DataFrame(
            {
                "Property": [
                    "E (Fixed)",
                    "ρ (Fixed)",
                    "ν (Fixed)",
                    "E (Free)",
                    "ρ (Free)",
                    "ν (Free)",
                ],
                "Value": [
                    f"{e_fixed:.6e} N/m²",
                    f"{rho_fixed:.6f} kg/m³",
                    f"{nu_fixed:.6f}",
                    f"{e_free:.6e} N/m²",
                    f"{rho_free:.6f} kg/m³",
                    f"{nu_free:.6f}",
                ],
            }
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

        reference_df = load_reference_dataset()

        if reference_df is not None and not reference_df.empty:
            validity_df = build_validity_table(
                input_df=input_df,
                reference_df=reference_df,
            )

            outside_count = int(
                (validity_df["Status"] == "Outside").sum()
            )

            if outside_count > 0:
                st.markdown(
                    """
                    <div class="warning-box">
                    Warning: One or more values are outside the
                    model-training range. The prediction involves
                    extrapolation and may be less reliable.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success(
                    "All input values are inside the training-data range."
                )

            with st.expander(
                "Show training validity-region table"
            ):
                st.dataframe(
                    validity_df,
                    use_container_width=True,
                    hide_index=True,
                )

        if VIDEO_PATH.exists():
            st.subheader("Axial Deformation")
            st.video(str(VIDEO_PATH))

    except Exception as error:
        st.error(f"Prediction error: {error}")


# =========================================================
# INITIAL INFORMATION
# =========================================================
else:
    st.markdown(
        """
        <div class="info-box">
        Enter the six material properties in the sidebar and
        select <strong>Predict Axial Frequency</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )