import streamlit as st
import pandas as pd
import joblib

from pandas.api.types import is_numeric_dtype


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   MAIN PAGE
   ========================================================= */

.stApp {
    background-color: #F5F7FB;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   TEXT COLORS
   ========================================================= */

h1, h2, h3, h4 {
    color: #172033 !important;
}

p {
    color: #475467 !important;
}

label {
    color: #344054 !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E4E7EC;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #172033 !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

div.stButton > button {
    width: 100%;
    min-height: 50px;

    background-color: #2563EB;
    color: white;

    border: none;
    border-radius: 10px;

    font-size: 16px;
    font-weight: 700;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E4E7EC;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border-color: #E4E7EC;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer-text {
    color: #98A2B3;
    font-size: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


# ============================================================
# LOAD MODEL + DATA
# ============================================================

try:

    model = load_model()
    train_data = load_data()

except Exception as e:

    st.error("❌ Could not load the model or dataset.")
    st.code(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏠 House Predictor")

    st.divider()

    st.subheader("How it works")

    st.markdown("""
    **1.** Enter property details

    **2.** Select the neighborhood

    **3.** Click **Predict House Price**

    **4.** View the estimated price
    """)

    st.divider()

    st.subheader("🤖 Model")

    st.info("Random Forest Regressor")

    st.subheader("📊 Dataset")

    st.write("House Prices Dataset")

    st.subheader("🎯 Target")

    st.write("SalePrice")


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🏠 House Price Predictor")

st.caption(
    "Estimate the expected sale price of a property "
    "using a Machine Learning model."
)

st.divider()


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.header("🏡 Property Details")

st.caption(
    "Enter the main characteristics of the property."
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2, col3 = st.columns(3, gap="large")


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    st.subheader("⭐ Quality & Size")

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=6,
        help="Overall material and finish quality."
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=300,
        max_value=6000,
        value=1500,
        step=50
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=5000,
        value=800,
        step=50
    )

    first_flr_sf = st.number_input(
        "1st Floor Area (sq ft)",
        min_value=300,
        max_value=4000,
        value=1000,
        step=50
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    st.subheader("🚗 Garage & Rooms")

    garage_cars = st.selectbox(
        "Garage Capacity",
        options=[0, 1, 2, 3, 4],
        index=2
    )

    garage_area = st.number_input(
        "Garage Area (sq ft)",
        min_value=0,
        max_value=1500,
        value=480,
        step=20
    )

    total_rooms = st.number_input(
        "Rooms Above Ground",
        min_value=1,
        max_value=15,
        value=6,
        step=1
    )

    full_bath = st.selectbox(
        "Full Bathrooms",
        options=[0, 1, 2, 3, 4],
        index=2
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    st.subheader("📅 Property Information")

    year_built = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2026,
        value=2000,
        step=1
    )

    year_remod = st.number_input(
        "Year Remodeled",
        min_value=1800,
        max_value=2026,
        value=2005,
        step=1
    )

    second_flr_sf = st.number_input(
        "2nd Floor Area (sq ft)",
        min_value=0,
        max_value=3000,
        value=500,
        step=50
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        options=sorted(
            train_data["Neighborhood"]
            .dropna()
            .unique()
        )
    )


# ============================================================
# PREDICTION SECTION
# ============================================================

st.divider()

st.header("🔮 Get Your Estimate")

st.caption(
    "Click the button below to calculate the estimated "
    "house price."
)


predict_button = st.button(
    "🚀 Predict House Price"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ====================================================
        # CREATE INPUT DATA
        # ====================================================

        input_data = {}

        feature_columns = (
            train_data
            .drop("SalePrice", axis=1)
            .columns
        )


        # ====================================================
        # FILL DEFAULT VALUES
        # ====================================================

        for column in feature_columns:

            if is_numeric_dtype(
                train_data[column]
            ):

                input_data[column] = (
                    train_data[column].median()
                )

            else:

                mode_values = (
                    train_data[column]
                    .dropna()
                    .mode()
                )

                if len(mode_values) > 0:

                    input_data[column] = (
                        mode_values.iloc[0]
                    )

                else:

                    input_data[column] = ""


        # ====================================================
        # REPLACE WITH USER INPUT
        # ====================================================

        input_data["OverallQual"] = overall_qual

        input_data["GrLivArea"] = gr_liv_area

        input_data["YearBuilt"] = year_built

        input_data["TotalBsmtSF"] = total_bsmt_sf

        input_data["GarageCars"] = garage_cars

        input_data["GarageArea"] = garage_area

        input_data["1stFlrSF"] = first_flr_sf

        input_data["FullBath"] = full_bath

        input_data["TotRmsAbvGrd"] = total_rooms

        input_data["2ndFlrSF"] = second_flr_sf

        input_data["YearRemodAdd"] = year_remod

        input_data["Neighborhood"] = neighborhood


        # ====================================================
        # CREATE DATAFRAME
        # ====================================================

        input_df = pd.DataFrame(
            [input_data]
        )


        # ====================================================
        # PREDICT
        # ====================================================

        with st.spinner(
            "🤖 Analyzing the property..."
        ):

            prediction = model.predict(
                input_df
            )[0]


        # ====================================================
        # RESULT
        # ====================================================

        st.success("✅ Prediction completed successfully!")

        st.subheader("🏠 Estimated Sale Price")

        st.metric(
            label="Predicted House Price",
            value=f"${prediction:,.0f}"
        )


        # ====================================================
        # PROPERTY SUMMARY
        # ====================================================

        st.divider()

        st.subheader("📊 Property Summary")

        summary1, summary2, summary3, summary4 = st.columns(4)


        with summary1:

            st.metric(
                label="⭐ Overall Quality",
                value=f"{overall_qual}/10"
            )


        with summary2:

            st.metric(
                label="🏠 Living Area",
                value=f"{gr_liv_area:,} sq ft"
            )


        with summary3:

            st.metric(
                label="📅 Year Built",
                value=str(year_built)
            )


        with summary4:

            st.metric(
                label="🚗 Garage Capacity",
                value=f"{garage_cars} Cars"
            )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.divider()

        st.subheader("🤖 Model Performance")

        performance1, performance2, performance3 = st.columns(3)


        with performance1:

            st.metric(
                label="Algorithm",
                value="Random Forest"
            )


        with performance2:

            st.metric(
                label="R² Score",
                value="0.794"
            )


        with performance3:

            st.metric(
                label="Test RMSE",
                value="$29.3K"
            )


        # ====================================================
        # ADDITIONAL INFORMATION
        # ====================================================

        st.divider()

        st.subheader("📌 Prediction Information")

        st.info(
            "This prediction is generated by a Random Forest "
            "regression model trained on the House Prices "
            "dataset. The displayed price is an estimate and "
            "not a guaranteed market value."
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏠 House Price Prediction By Yash• "
    "Random Forest Regression • "
    "Python + Scikit-learn + Streamlit"
)