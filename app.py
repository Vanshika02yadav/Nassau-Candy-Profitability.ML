import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Nassau Candy Profitability Analysis",
    page_icon="🍬",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

DATA_FILE = "Nassau Candy Distributor.csv"


@st.cache_data
def load_data():

    df = pd.read_csv(DATA_FILE)

    # Convert dates
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Calculate margin
    df["Margin %"] = np.where(
        df["Sales"] != 0,
        (df["Gross Profit"] / df["Sales"]) * 100,
        0
    )

    return df


# =========================================================
# ERROR HANDLING
# =========================================================

try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "CSV file not found. Make sure 'Nassau Candy Distributor.csv' "
        "is in the same folder as app.py."
    )

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🍬 Nassau Candy Distributor")

st.subheader(
    "Product Line Profitability & Margin Performance Analysis"
)

st.write(
    "This machine learning project analyzes product profitability, "
    "sales performance and margin performance and predicts whether "
    "a transaction is likely to have a High Margin or Low Margin."
)


# =========================================================
# KEY PERFORMANCE INDICATORS
# =========================================================

total_orders = df["Order ID"].nunique()

total_sales = df["Sales"].sum()

total_profit = df["Gross Profit"].sum()

overall_margin = (
    total_profit / total_sales
) * 100


st.markdown("## 📊 Business Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )


with col2:

    st.metric(
        "Total Sales",
        f"{total_sales:,.2f}"
    )


with col3:

    st.metric(
        "Gross Profit",
        f"{total_profit:,.2f}"
    )


with col4:

    st.metric(
        "Overall Margin",
        f"{overall_margin:.2f}%"
    )


st.divider()


# =========================================================
# EXPLORATORY DATA ANALYSIS
# =========================================================

st.header("📈 Profitability Analysis")


tab1, tab2, tab3 = st.tabs(
    [
        "Product Analysis",
        "Division Analysis",
        "Regional Analysis"
    ]
)


# =========================================================
# PRODUCT ANALYSIS
# =========================================================

with tab1:

    product = df.groupby(
        "Product Name"
    ).agg(

        Orders=("Order ID", "count"),

        Sales=("Sales", "sum"),

        Units=("Units", "sum"),

        Gross_Profit=("Gross Profit", "sum")

    ).reset_index()


    product["Margin %"] = (
        product["Gross_Profit"]
        / product["Sales"]
    ) * 100


    st.subheader("Product Profitability")

    st.dataframe(
        product.sort_values(
            "Margin %",
            ascending=False
        ),
        use_container_width=True
    )


    # Top 15 products
    top_products = (
        product
        .sort_values("Margin %")
        .tail(15)
    )


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    ax.barh(
        top_products["Product Name"],
        top_products["Margin %"]
    )


    ax.set_xlabel("Margin (%)")

    ax.set_title(
        "Top 15 Product Margin Performance"
    )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# DIVISION ANALYSIS
# =========================================================

with tab2:

    division = df.groupby(
        "Division"
    ).agg(

        Sales=("Sales", "sum"),

        Units=("Units", "sum"),

        Gross_Profit=("Gross Profit", "sum")

    ).reset_index()


    division["Margin %"] = (
        division["Gross_Profit"]
        / division["Sales"]
    ) * 100


    st.subheader("Division Performance")

    st.dataframe(
        division,
        use_container_width=True
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        division["Division"],
        division["Margin %"]
    )


    ax.set_ylabel("Margin (%)")

    ax.set_title(
        "Margin Performance by Division"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# REGIONAL ANALYSIS
# =========================================================

with tab3:

    region = df.groupby(
        "Region"
    ).agg(

        Sales=("Sales", "sum"),

        Units=("Units", "sum"),

        Gross_Profit=("Gross Profit", "sum")

    ).reset_index()


    region["Margin %"] = (
        region["Gross_Profit"]
        / region["Sales"]
    ) * 100


    st.subheader("Regional Performance")

    st.dataframe(
        region,
        use_container_width=True
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        region["Region"],
        region["Margin %"]
    )


    ax.set_ylabel("Margin (%)")

    ax.set_title(
        "Margin Performance by Region"
    )

    ax.tick_params(
        axis="x",
        rotation=45
    )


    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# MACHINE LEARNING
# =========================================================

st.divider()

st.header("🤖 Machine Learning Prediction")


st.write(
    "The machine learning model uses a Random Forest Classifier "
    "to predict whether a transaction belongs to the High Margin "
    "or Low Margin category."
)


# =========================================================
# FEATURE ENGINEERING
# =========================================================

data = df.sort_values(
    "Order Date"
).copy()


data["year"] = (
    data["Order Date"].dt.year
)


data["month"] = (
    data["Order Date"].dt.month
)


data["quarter"] = (
    data["Order Date"].dt.quarter
)


data["dayofweek"] = (
    data["Order Date"].dt.dayofweek
)


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split_date = data[
    "Order Date"
].quantile(0.80)


train = data[
    data["Order Date"] <= split_date
].copy()


test = data[
    data["Order Date"] > split_date
].copy()


# =========================================================
# TARGET CREATION
# =========================================================

margin_threshold = (
    train["Margin %"].median()
)


train["High_Margin"] = (
    train["Margin %"]
    >= margin_threshold
).astype(int)


test["High_Margin"] = (
    test["Margin %"]
    >= margin_threshold
).astype(int)


# =========================================================
# FEATURES
# =========================================================

features = [

    "Ship Mode",

    "Country/Region",

    "Division",

    "Region",

    "Units",

    "year",

    "month",

    "quarter",

    "dayofweek",

    "Product Name"

]


X_train = train[features]

y_train = train["High_Margin"]


X_test = test[features]

y_test = test["High_Margin"]


# Categorical columns

categorical = [

    column

    for column in features

    if X_train[column].dtype == "object"

]


# Numerical columns

numeric = [

    column

    for column in features

    if column not in categorical

]


# =========================================================
# PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical
        ),

        (
            "numerical",

            "passthrough",

            numeric
        )

    ]

)


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "classifier",

            RandomForestClassifier(

                n_estimators=300,

                max_depth=12,

                min_samples_leaf=3,

                class_weight="balanced",

                random_state=42,

                n_jobs=-1
            )
        )

    ]

)


# Train model

model.fit(
    X_train,
    y_train
)


# =========================================================
# MODEL EVALUATION
# =========================================================

predictions = model.predict(
    X_test
)


probabilities = model.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)


roc_auc = roc_auc_score(
    y_test,
    probabilities
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    st.metric(
        "ROC-AUC",
        f"{roc_auc:.4f}"
    )


st.info(
    f"High Margin threshold: {margin_threshold:.2f}%"
)


# =========================================================
# PREDICTION FORM
# =========================================================

st.subheader(
    "🔮 Predict a New Transaction"
)


col1, col2 = st.columns(2)


with col1:

    ship_mode = st.selectbox(
        "Ship Mode",
        sorted(
            df["Ship Mode"]
            .dropna()
            .unique()
        )
    )


    country = st.selectbox(
        "Country/Region",
        sorted(
            df["Country/Region"]
            .dropna()
            .unique()
        )
    )


    division_name = st.selectbox(
        "Division",
        sorted(
            df["Division"]
            .dropna()
            .unique()
        )
    )


    region_name = st.selectbox(
        "Region",
        sorted(
            df["Region"]
            .dropna()
            .unique()
        )
    )


    product_name = st.selectbox(
        "Product Name",
        sorted(
            df["Product Name"]
            .dropna()
            .unique()
        )
    )


with col2:

    units = st.number_input(
        "Units",
        min_value=1,
        value=4,
        step=1
    )


    year = st.number_input(
        "Year",
        min_value=2000,
        max_value=2100,
        value=2025
    )


    month = st.slider(
        "Month",
        min_value=1,
        max_value=12,
        value=6
    )


    dayofweek = st.slider(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=2
    )


quarter = (
    (month - 1) // 3
) + 1


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Margin",
    type="primary"
):

    new_transaction = pd.DataFrame(

        [{

            "Ship Mode": ship_mode,

            "Country/Region": country,

            "Division": division_name,

            "Region": region_name,

            "Units": units,

            "year": year,

            "month": month,

            "quarter": quarter,

            "dayofweek": dayofweek,

            "Product Name": product_name

        }]

    )


    prediction = model.predict(
        new_transaction
    )[0]


    probability = model.predict_proba(
        new_transaction
    )[0, 1]


    if prediction == 1:

        st.success(
            f"✅ HIGH MARGIN\n\n"
            f"Probability of High Margin: "
            f"{probability * 100:.2f}%"
        )

    else:

        st.warning(
            f"⚠️ LOW MARGIN\n\n"
            f"Probability of Low Margin: "
            f"{(1 - probability) * 100:.2f}%"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Built with Python • Pandas • NumPy • Matplotlib • "
    "Scikit-learn • Streamlit"
)