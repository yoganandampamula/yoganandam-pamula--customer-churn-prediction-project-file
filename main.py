import streamlit as st
import pandas as pd
from PIL import Image
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from preprocessing import preprocess
import plotly.express as px
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model from disk
model = joblib.load(r"./Models/model.sav")

def main():
    # Setting Application title
    st.title('Telco Customer Churn Prediction App')

    # Setting Application description
    st.markdown("""
        :dart:  This Streamlit app is made to predict customer churn in a fictional telecommunication use case.
        The application is functional for both online prediction and batch data prediction.
        """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('app.jpg')
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    # Online Prediction
    st.info("Input data below")
    st.subheader("Demographic data")
    seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'), index=1)
    dependents = st.selectbox('Dependent:', ('Yes', 'No'), index=0)
    st.subheader("Payment data")
    tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
    contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two years'), index=0)
    paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'), index=0)
    PaymentMethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'), index=0)
    monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
    totalcharges = st.number_input('The total amount charged to the customer', min_value=0, max_value=10000, value=0)

    st.subheader("Services signed up for")
    mutliplelines = st.selectbox("Does the customer have multiple lines", ('Yes', 'No', 'No phone service'), index=0)
    phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'), index=0)
    internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'), index=1)
    onlinesecurity = st.selectbox("Does the customer have online security", ('Yes', 'No', 'No internet service'), index=0)
    onlinebackup = st.selectbox("Does the customer have online backup", ('Yes', 'No', 'No internet service'), index=0)
    techsupport = st.selectbox("Does the customer have technology support", ('Yes', 'No', 'No internet service'), index=0)
    streamingtv = st.selectbox("Does the customer stream TV", ('Yes', 'No', 'No internet service'), index=0)
    streamingmovies = st.selectbox("Does the customer stream movies", ('Yes', 'No', 'No internet service'), index=0)

    data = {
        'SeniorCitizen': seniorcitizen,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phoneservice,
        'MultipleLines': mutliplelines,
        'InternetService': internetservice,
        'OnlineSecurity': onlinesecurity,
        'OnlineBackup': onlinebackup,
        'TechSupport': techsupport,
        'StreamingTV': streamingtv,
        'StreamingMovies': streamingmovies,
        'Contract': contract,
        'PaperlessBilling': paperlessbilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': monthlycharges,
        'TotalCharges': totalcharges
    }
    features_df = pd.DataFrame.from_dict([data])
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.write('Overview of input is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(features_df)

    # Preprocess inputs
    preprocess_df = preprocess(features_df, 'Online')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    preprocess_df_imputed = pd.DataFrame(imputer.fit_transform(preprocess_df), columns=preprocess_df.columns)

    prediction = model.predict(preprocess_df_imputed)

    if st.button('Predict Online', key='predict_online_button'):
        if prediction == 1:
            st.warning('Yes, the customer will terminate the service.')
        else:
            st.success('No, the customer is happy with Telco Services.')

    # Batch Prediction
    st.subheader("Dataset upload")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Get overview of data
        st.write(data.head())
        st.markdown("<h3></h3>", unsafe_allow_html=True)

        # Preprocess inputs
        preprocess_df = preprocess(data, "Batch")

        # Handle missing values
        preprocess_df_imputed = pd.DataFrame(imputer.transform(preprocess_df), columns=preprocess_df.columns)

        if st.button('Predict Batch', key='predict_batch_button'):
            # Get batch prediction
            prediction = model.predict(preprocess_df_imputed)
            prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
            prediction_df = prediction_df.replace({
                1: 'Yes, the customer will terminate the service.',
                0: 'No, the customer is happy with Telco Services.'
            })

            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.subheader('Prediction')
            st.write(prediction_df)

            # Add a bar chart to show the distribution of predicted churn and non-churn
            fig = px.bar(prediction_df["Predictions"].value_counts(), x=prediction_df["Predictions"].value_counts().index, y=prediction_df["Predictions"].value_counts(), labels={'x': 'Prediction', 'y': 'Count'})
            fig.update_layout(title_text='Predicted Churn Distribution')
            st.plotly_chart(fig)

            # Additional Visualizations
            st.subheader("Additional Visualizations")

            # Pie Chart
            st.write("### Pie Chart - Distribution of Predictions")
            fig_pie = px.pie(prediction_df, names="Predictions", title='Distribution of Predictions')
            st.plotly_chart(fig_pie)

            # Correlation Matrix and Heatmap
            st.write("### Correlation Matrix and Heatmap")
            corr_matrix = preprocess_df_imputed.corr()
            sns.set_theme(style="white")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
            st.pyplot(fig_corr)
            # Box Plot
            st.write("### Box Plot - Distribution of Numerical Features")
            fig_box = px.box(preprocess_df_imputed, y=preprocess_df_imputed.columns, title='Distribution of Numerical Features')
            st.plotly_chart(fig_box)

if __name__ == '__main__':
    main()
