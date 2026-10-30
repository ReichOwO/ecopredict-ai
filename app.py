
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="EcoPredict AI",
    page_icon="🌱"
)

st.title("🌱 EcoPredict AI")
st.subheader("University Cafeteria Meal Demand Forecasting")

st.write(
    "Predict meal demand and support sustainable "
    "food preparation using machine learning."
)

uploaded_file = st.file_uploader(
    "Upload cafeteria sales CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required = ["menu_item", "weekday", "meals_sold"]

    if not all(col in df.columns for col in required):
        st.error("Missing required CSV columns.")
        st.stop()

    st.success("File uploaded successfully!")

    st.write("Historical Sales Data")
    st.dataframe(df.head())

    X = pd.get_dummies(df[["menu_item", "weekday"]])
    y = df["meals_sold"]

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    st.success("AI model trained successfully!")

    menu = st.selectbox(
        "Choose a menu",
        sorted(df["menu_item"].unique())
    )

    weekday = st.selectbox(
        "Choose a weekday",
        ["Monday", "Tuesday", "Wednesday",
         "Thursday", "Friday", "Saturday", "Sunday"]
    )

    if st.button("Predict Demand"):
        day_number = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ].index(weekday)

        new_data = pd.DataFrame([{
            "menu_item": menu,
            "weekday": day_number
        }])

        new_X = pd.get_dummies(new_data)
        new_X = new_X.reindex(
            columns=X.columns,
            fill_value=0
        )

        prediction = model.predict(new_X)[0]

        st.metric(
            "Predicted Meals",
            round(max(0, prediction))
        )

        st.info(
            "This is a prototype prediction. "
            "Vendors should review results."
        )

else:
    st.info("Upload a CSV file to start.")
