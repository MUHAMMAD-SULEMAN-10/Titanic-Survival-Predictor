import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None

# Title
st.markdown('<h1 class="main-header">🚢 Titanic Survival Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/300px-RMS_Titanic_3.jpg", use_container_width=True)
    page = st.radio("Select Page:",
                    ["📊 Data Explorer",
                     "🤖 Model Training",
                     "🔮 Make Predictions",
                     "📈 Model Performance"])
    st.markdown("---")
    if st.button("♻️ Reset Data"):
        st.session_state.train_data = None
        st.session_state.test_data = None
        st.session_state.model = None
        st.experimental_rerun()
    st.info("💡 Upload your Titanic CSV files to explore, train, and predict survival.")

# Helper Functions
@st.cache_data
def load_default_data():
    """Load default Titanic data if no upload is provided"""
    try:
        train_df = pd.read_csv('train.csv')
        test_df = pd.read_csv('titanic_test.csv')
        return train_df, test_df
    except:
        st.error("⚠️ Please make sure 'train.csv' and 'titanic_test.csv' are available.")
        return None, None

def preprocess_data(train_df, test_df):
    """Clean and encode Titanic data"""
    train_data = train_df.copy()
    test_data = test_df.copy()

    # Fill missing values
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    test_data['Age'].fillna(train_data['Age'].median(), inplace=True)

    most_common_port = train_data['Embarked'].mode()[0]
    train_data['Embarked'].fillna(most_common_port, inplace=True)
    test_data['Embarked'].fillna(most_common_port, inplace=True)

    test_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

    # Encode categorical columns
    train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
    test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
    train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    return train_data, test_data
# --- PAGE 1: DATA EXPLORER ---
if page == "📊 Data Explorer":
    st.markdown('<h2 class="sub-header">📊 Data Explorer</h2>', unsafe_allow_html=True)

    # File uploaders
    uploaded_train = st.file_uploader("Upload Training Data (train.csv)", type=['csv'])
    uploaded_test = st.file_uploader("Upload Test Data (test.csv)", type=['csv'])

    # If user uploads both files → read and store in session
    if uploaded_train is not None and uploaded_test is not None:
        train_df = pd.read_csv(uploaded_train)
        test_df = pd.read_csv(uploaded_test)
        st.session_state.train_data = train_df
        st.session_state.test_data = test_df
        st.success("✅ Uploaded data loaded successfully!")

        # Refresh app to show new uploaded data
        st.rerun()

    # If no upload yet → load default data once
    elif st.session_state.get("train_data") is None or st.session_state.get("test_data") is None:
        train_df, test_df = load_default_data()
        if train_df is not None:
            st.session_state.train_data = train_df
            st.session_state.test_data = test_df

    # Use whichever data is available in session
    if st.session_state.get("train_data") is not None:
        df = st.session_state.train_data

        st.markdown("### 🔍 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        # --- Summary Metrics ---
        st.markdown("### 📊 Data Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Passengers", len(df))
        with col2:
            if 'Survived' in df.columns:
                survivors = df['Survived'].sum()
                st.metric("Survivors", f"{survivors} ({survivors / len(df) * 100:.1f}%)")
            else:
                st.metric("Survivors", "N/A")
        with col3:
            if 'Survived' in df.columns:
                deaths = len(df) - survivors
                st.metric("Deaths", f"{deaths} ({deaths / len(df) * 100:.1f}%)")
            else:
                st.metric("Deaths", "N/A")

        # --- Age Distribution Plot ---
        if 'Age' in df.columns:
            st.markdown("### 📈 Age Distribution")
            fig = px.histogram(df, x='Age', nbins=30, title='Age Distribution', color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Column 'Age' not found in uploaded data.")

# --- PAGE 2: MODEL TRAINING ---
elif page == "🤖 Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)

    if st.session_state.train_data is None:
        st.warning("⚠️ Please upload or load data first from the Data Explorer page.")
    else:
        train_df, test_df = preprocess_data(
            st.session_state.train_data, st.session_state.test_data
        )

        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        test_size = st.slider("Test Split %", 10, 40, 20, 5) / 100

        if st.button("🚀 Train Model", type="primary"):
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            X = train_df[features]
            y = train_df['Survived']

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)

            st.session_state.model = model
            st.session_state.features = features
            st.success(f"✅ Model trained successfully with {accuracy:.2%} accuracy!")

            st.metric("Training Samples", len(X_train))
            st.metric("Validation Samples", len(X_val))

            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title='Feature Importance', color='Importance', color_continuous_scale='blues')
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 3: MAKE PREDICTIONS ---
elif page == "🔮 Make Predictions":
    st.markdown('<h2 class="sub-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first from the Model Training page.")
    else:
        st.info("Enter passenger details to predict survival:")

        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3])
            sex = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 0, 80, 30)
            sibsp = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
        with col2:
            parch = st.number_input("Parents/Children Aboard", 0, 6, 0)
            fare = st.number_input("Fare ($)", 0.0, 512.0, 32.0, 0.1)
            embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

        if st.button("🔮 Predict Survival", type="primary"):
            sex_encoded = 1 if sex == "Male" else 0
            embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
            embarked_encoded = embarked_map[embarked]

            input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
            prediction = st.session_state.model.predict(input_data)[0]
            probability = st.session_state.model.predict_proba(input_data)[0]

            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2>✅ SURVIVED</h2>
                        <p>Survival Probability: {probability[1]:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                        <h2>❌ DID NOT SURVIVE</h2>
                        <p>Survival Probability: {probability[1]:.1%}</p>
                    </div>
                """, unsafe_allow_html=True)

# --- PAGE 4: MODEL PERFORMANCE ---
elif page == "📈 Model Performance":
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first!")
    else:
        st.success("🎯 Model is trained and ready for predictions!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Type", "Random Forest")
            st.metric("Number of Features", 7)
        with col2:
            st.metric("Status", "✅ Ready")
            st.metric("Trees", 100)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚢 Titanic Survival Predictor | Built with Streamlit</p>
        <p>Data Science Project | Machine Learning Classification</p>
    </div>
""", unsafe_allow_html=True)
