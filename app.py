import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import io
import os
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import seaborn as sns
import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc

# Function to load models
# Function to load models
def load_models():
    models = {}
 
    try:
        with open('final_model', 'rb') as file:
            models['Random Forest'] = pickle.load(file)
        print("Random Forest model loaded successfully!")
    except Exception as e:
        print(f"Error loading Random Forest model: {e}")

 

    return models
df = pd.read_csv("data.csv")

# Sidebar navigation using option_menu
with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=['Home', 'Data', 'Dashboard', 'Prediction', 'History'],
        icons=['house', 'table', 'bar-chart-line', 'lightning', 'clock-history'],
        orientation="vertical",
        default_index=0  # Optionally set default index
    )

# Functions for different pages
def home_page():
    st.markdown('<h2 style="text-align: center;">Fraud Detection Prediction Application</h2>', unsafe_allow_html=True)
    
    st.write("""Welcome to the Fraud Detection Dashboard, your go-to solution for identifying 
             and mitigating fraudulent transactions. This tool harnesses the power of machine 
             learning to provide real-time insights and ensure the security of your financial systems.""")
    st.image("static/2.png", caption="Fraud Detection Prediction", use_column_width=True)

def data_page():
    st.markdown('<h2 style="text-align: center;">Data</h2>', unsafe_allow_html=True)
    st.write("This page allows users to explore and analyze the dataset.")
    try:
        df = pd.read_csv("data.csv")
        st.write("### Dataset Overview")
        st.write("#### First 5 rows")
        st.dataframe(df.head())
        if st.checkbox("Show Data Info"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
        if st.checkbox("Show Summary Statistics"):
            st.write("#### Summary Statistics")
            st.write(df.describe(include='all'))
        column = st.selectbox("Select a column to view its distribution:", df.columns)
        if column:
            st.write(f"#### Distribution of {column}")
            st.write(df[column].value_counts())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

def bar_plot_with_annotations():
    left_counts = df['isfraud'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(left_counts.index, left_counts.values, color=sns.color_palette('viridis', len(left_counts)))
    for p in bars:
        height = p.get_height()
        ax.annotate(f'{height}',
                    xy=(p.get_x() + p.get_width() / 2., height),
                    xytext=(0, -10),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    color='white',
                    fontsize=10)
    ax.set_title('Bar Plot of isfraud Distribution')
    #ax.set_xlabel('Left (0 = Stayed, 1 = Left)')
    ax.set_ylabel('Count')
    ax.set_xticks([])
    ax.legend(bars, ['0', '1'], title="Left", loc='upper right')
    return fig

def heatmap_plot():
    numeric_df = df.select_dtypes(include=[float, int])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                linewidths=0.5, linecolor='black', fmt='.2f', ax=ax)
    ax.set_title('Correlation Heatmap')
    return fig
def plot_feature_distribution(feature):
    plt.figure(figsize=(8, 5))
    
    # Define the list of features you want to check against
    special_features = ['productcd']

    # Check if the selected feature is categorical OR part of the special_features list
    if df[feature].dtype == 'object' or feature in special_features:  # Categorical feature
        ax = sns.countplot(
            data=df,
            x=feature,
            hue="isfraud",
            palette='viridis',
            edgecolor='gray'
        )

        # Add bar labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45)

        # Add title and axis labels
        plt.title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Customize legend
        if len(df[feature].unique()) > 1:  # Only add legend if there are multiple categories
            plt.legend(title='isfraud', loc='upper right')

        # Add grid lines to the y-axis for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Adjust layout to avoid overlap
        plt.tight_layout()

        # Show plot
        st.pyplot(plt.gcf())
    else:  # Numeric feature
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.histplot(df[df['isfraud'] == 0][feature], kde=True, color='skyblue', label='Not Fraud', bins=30, alpha=0.6, ax=ax)
        sns.histplot(df[df['isfraud'] == 1][feature], kde=True, color='salmon', label='isfraud', bins=30, alpha=0.6, ax=ax)
        
        plt.title(f'Distribution of {feature}', fontsize=14, fontweight='bold')
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)


def dashboard_page():
    #st.markdown('<h2 style="text-align: center;">Dashboard</h2>', unsafe_allow_html=True)



    # Check the selected dashboard view
    
        
        st.markdown('<h3 style="text-align: center;">Feature Analysis Dashboard</h3>', unsafe_allow_html=True)
        feature_analysis_view = option_menu(
            menu_title='Feature Analysis Sections',
            options=['Target Feature', 'Other Features'],
            icons=['target', 'list'],
            orientation='horizontal'
        )

        if feature_analysis_view == "Target Feature":
            #st.write("Analysis of Target (left) Feature")
            fig1 = bar_plot_with_annotations()
            st.pyplot(fig1)
            st.markdown("---")
            fig2 = heatmap_plot()
            st.pyplot(fig2)

        elif feature_analysis_view == "Other Features":
            #st.write("Analysis of Other Features")
            # Dropdown for feature selection
            feature = st.selectbox("Select Feature to Plot", [col for col in df.columns if col != 'isfraud'])
            if feature:
                plot_feature_distribution(feature)
    
    

        
def get_range_of_values(column_name):
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    return min_value, max_value

def get_matching_records(features_list):
    return pd.DataFrame([features_list])


# Path to the history CSV file
csv_file = "prediction_history.csv"

# Initialize prediction history in session state if it doesn't exist
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame()  # Initialize as empty DataFrame

# Load prediction history from CSV if it exists
if os.path.exists(csv_file):
    st.session_state.prediction_history = pd.read_csv(csv_file)
else:
    st.session_state.prediction_history = pd.DataFrame()  # Ensure it's an empty DataFrame if the file doesn't exist

# Function to add a new record to the history
def add_to_history(new_record_df):
    if isinstance(new_record_df, pd.DataFrame):
        # Ensure that the new_record_df has the same columns as st.session_state.prediction_history
        if not st.session_state.prediction_history.empty:
            new_record_df = new_record_df.reindex(columns=st.session_state.prediction_history.columns)
        
        # Append new record to the session history DataFrame
        st.session_state.prediction_history = pd.concat(
            [st.session_state.prediction_history, new_record_df],
            ignore_index=True,
            sort=False
        )

        # Save updated DataFrame to CSV
        st.session_state.prediction_history.to_csv(csv_file, index=False)
        #st.write("New record added and saved to CSV.")
    else:
        st.write("Error: new_record_df is not a DataFrame.")


def make_predictions(model, features_df):
    """Function to make predictions and get probabilities from a model."""
    try:
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(features_df)
            prediction = model.predict(features_df)
            prediction_result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
            proba_result = pred_proba[0, 1] * 100 if prediction[0] == 1 else pred_proba[0, 0] * 100
            return prediction_result, f"{proba_result:.2f} %"
        else:
            prediction = model.predict(features_df)
            prediction_result = 'Fraud' if prediction[0] == 1 else 'Not Frauf'
            return prediction_result, "N/A"
    except Exception as e:
        return 'Error', str(e)
def prediction_page():
    st.markdown('<h2 style="text-align: center;">Fraud Detection Prediction</h2>', unsafe_allow_html=True)
    
    
    # Initialize session state for predictions
    if 'model_predictions' not in st.session_state:
        st.session_state.model_predictions = []

    # Initialize the features list
    features_list = {}

    # Collect input features from the user
    st.subheader("Transaction Features")
    # Define dropdown for the object column and add it to the features_list
    features_list['productcd'] = st.selectbox('ProductCD', options=['W', 'C', 'H', 'S', 'R'])
    # Define input fields for numeric columns and add them to the features_list
    features_list['c12'] = st.number_input('c12', value=0.0)
    features_list['c8'] = st.number_input('c8', value=0.0)
    features_list['c7'] = st.number_input('c7', value=0.0)
    features_list['v317'] = st.number_input('v317', value=0.0)
    features_list['c4'] = st.number_input('c4', value=0.0)
    features_list['c13'] = st.number_input('c13', value=0.0)
    features_list['c14'] = st.number_input('c14', value=0.0)
    features_list['c10'] = st.number_input('c10', value=0.0)
    features_list['card3'] = st.number_input('card3', value=0.0)
    features_list['c5'] = st.number_input('c5', value=0.0)
    features_list['v302'] = st.number_input('v302', value=0.0)
    features_list['v283'] = st.number_input('v283', value=0.0)
    features_list['c1'] = st.number_input('c1', value=0.0)
    features_list['v294'] = st.number_input('v294', value=0.0)
    features_list['v318'] = st.number_input('v318', value=0.0)
    features_list['v308'] = st.number_input('v308', value=0.0)
    features_list['v303'] = st.number_input('v303', value=0.0)
    features_list['v133'] = st.number_input('v133', value=0.0)
    features_list['v304'] = st.number_input('v304', value=0.0)




    # Display the input features in a table
    features_df = pd.DataFrame([features_list])
    st.subheader("Input Features")
    st.dataframe(features_df)

    # Load the models
    models = load_models()
    rf_model = models.get('Random Forest')

    if st.button('Predict Fraud'):
        if rf_model:
            # Make the prediction with Random Forest
            rf_prediction_result, rf_proba_result = make_predictions(rf_model, features_df)

            # Add the prediction result and probability to the features_df
            features_df['Prediction'] = rf_prediction_result
            features_df['Probability'] = rf_proba_result

            # Store the result in prediction history
            add_to_history(features_df)
            # Display the Random Forest results with highlighted style
            background_color = "#f8d7da" if rf_prediction_result == "Fraud" else "#d4edda"  # Red for fraud, green otherwise
            text_color = "#721c24" if rf_prediction_result == "Fraud" else "#155724"  # Dark red for fraud text, green otherwise
            
            st.markdown(f"""
            <div style="background-color: {background_color}; text-align: center; padding: 10px; border-radius: 5px; border: 1px solid {background_color};">
            <strong>Prediction Result:</strong> <span style="color: {text_color};">{rf_prediction_result}</span><br>
            <strong>Probability Result:</strong> <span style="color: {text_color};">{rf_proba_result}</span>
            </div>
            """, unsafe_allow_html=True)

                
##HISTORY            
def history_page():
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem; /* Adjust this value to control the top margin */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown('<h2 style="text-align: center;">History</h2>', unsafe_allow_html=True)
    st.write("Review the history of previous predictions or data analyses.")

    # Path to the history CSV file
    csv_file = "prediction_history.csv"
    
    # Check if the file exists
    if os.path.exists(csv_file):
        history_df = pd.read_csv(csv_file)
        if not history_df.empty:
            st.dataframe(history_df)
        else:
            st.write("No history found in the CSV file.")
    else:
        st.write("No history file found.")
    

# Routing logic based on the selected menu option
if selected == "Home":
    home_page()
    # Example of getting the probabilities
    
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.1rem; /* Adjust this value to control the top margin */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        <div class="team-section">
            <h3 align=center>Development Team</h3>
            <p >Aysegül, Damla,Esra, Emre, Ezgi</p>
            <p align=center>Built with <span class="heart">❤️</span></p>
        </div>
        """, unsafe_allow_html=True)
    

    

elif selected == "Data":
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem; /* Adjust this value to control the top margin */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    data_page()
elif selected == "Dashboard":
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem; /* Adjust this value to control the top margin */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    dashboard_page()
elif selected == "Prediction":
    st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem; /* Adjust this value to control the top margin */
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    prediction_page()
elif selected == "History":
    history_page()

# Inject custom CSS (optional)
st.markdown(
    """
    <style>
    .team-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .team-section h3 {
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .team-section p {
        font-size: 1.1rem;
        color: #7f8c8d;
    }
    .heart {
        color: #e74c3c;
    }
    .active {
        background-color: #e74c3c;
        color: white;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton {
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
