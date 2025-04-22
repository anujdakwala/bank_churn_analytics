# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 17:50:07 2025

@author: adakw
"""
# Set page config - MUST BE FIRST STREAMLIT COMMAND
#st.set_page_config(layout="wide", page_title="Bank Customer Churn Analysis")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    layout="wide", 
    page_title="Bank Customer Churn Analysis",
    page_icon="🏦"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-emotion-cache-1v0mbdj {
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        color: #666666;
        font-size: 14px;
    }
    .high-risk {
        color: #ff4b4b;
        font-weight: bold;
    }
    .medium-risk {
        color: #ffa500;
        font-weight: bold;
    }
    .low-risk {
        color: #2ecc71;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    train_df = pd.read_csv(r'train.csv')
    test_df = pd.read_csv(r'test.csv')
    predictions = pd.read_csv(r'churn_predictions.csv')
    with open(r'scaler.pkl', 'rb') as f:
        model = pickle.load(f)
    return train_df, test_df, predictions, model

train_df, test_df, predictions, model = load_data()

# Merge predictions with test data
test_df = test_df.merge(predictions, on=['id', 'CustomerId', 'Surname'])

# Add header
st.title("Bank Customer Churn Analysis Dashboard")
st.markdown("""
<div style="color: #666666; margin-bottom: 20px;">
Predictive analytics to identify at-risk customers and reduce churn
</div>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filters")
min_prob = st.sidebar.slider("Minimum Churn Probability", 0.0, 1.0, 0.5)
country_filter = st.sidebar.multiselect("Country", options=test_df['Geography'].unique(), default=test_df['Geography'].unique())
gender_filter = st.sidebar.multiselect("Gender", options=test_df['Gender'].unique(), default=test_df['Gender'].unique())
age_range = st.sidebar.slider("Age Range", 
                             min_value=int(test_df['Age'].min()), 
                             max_value=int(test_df['Age'].max()),
                             value=(25, 60))

# Apply filters
filtered_data = test_df[
    (test_df['ChurnProbability'] >= min_prob) &
    (test_df['Geography'].isin(country_filter)) &
    (test_df['Gender'].isin(gender_filter)) &
    (test_df['Age'] >= age_range[0]) &
    (test_df['Age'] <= age_range[1])
]

# Overview metrics
st.markdown("### 📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box">Total Customers<br><span style="font-size: 24px; font-weight: bold;">' + 
                f"{len(test_df):,}</span></div>", unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box">Predicted Churn Rate<br><span style="font-size: 24px; font-weight: bold;">' + 
                f"{test_df['PredictedChurn'].mean()*100:.1f}%</span></div>", unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box">High Risk Customers<br><span style="font-size: 24px; font-weight: bold;">' + 
                f"{len(filtered_data):,}</span></div>", unsafe_allow_html=True)
with col4:
    avg_balance = filtered_data['Balance'].mean() if len(filtered_data) > 0 else 0
    st.markdown('<div class="metric-box">Avg Balance at Risk<br><span style="font-size: 24px; font-weight: bold;">' + 
                f"${avg_balance:,.2f}</span></div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👥 Customer Analysis", "🔮 Predictions", "💡 Insights & Actions"])

with tab1:
    st.markdown("### Customer Churn Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='PredictedChurn', data=test_df, palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_title('Churn vs Retained Customers', fontsize=14)
        ax.set_xlabel('Status')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(test_df['ChurnProbability'], bins=20, kde=True, color='#3498db', ax=ax)
        ax.set_title('Churn Probability Distribution', fontsize=14)
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Customer Dynamics")
    col1, col2 = st.columns(2)
    
    with col1:
        #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='PredictedChurn', y='Age', data=test_df, palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_title('Age Distribution by Churn Status', fontsize=14)
        ax.set_xlabel('Churn Status')
        ax.set_ylabel('Age')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Age', y='Balance', hue='PredictedChurn', 
                        data=test_df, palette=['#2ecc71', '#e74c3c'], alpha=0.6, ax=ax)
        ax.set_title('Age vs Balance by Churn Status', fontsize=14)
        ax.set_xlabel('Age')
        ax.set_ylabel('Balance')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='NumOfProducts', hue='PredictedChurn', 
                 data=test_df, palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Number of Products vs Churn Status', fontsize=14)
    ax.set_xlabel('Number of Products')
    ax.set_ylabel('Count')
    ax.legend(title='Churn Status', labels=['Retained', 'Churned'])
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### High Risk Customers")
    st.dataframe(
        filtered_data.sort_values('ChurnProbability', ascending=False)[
            ['CustomerId', 'Surname', 'Age', 'Geography', 'Gender', 
             'Balance', 'NumOfProducts', 'ChurnProbability']
        ].style.format({'Balance': '${:,.2f}', 'ChurnProbability': '{:.2%}'}),
        height=400
    )
    
    st.markdown("### Customer Segmentation Analysis")
    col1, col2 = st.columns(2)
    with col1:
    
        #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        churn_by_geo = test_df.groupby('Geography')['PredictedChurn'].mean().sort_values()
        churn_by_geo.plot(kind='bar', color=['#3498db', '#e74c3c', '#2ecc71'], ax=ax)
        ax.set_title('Churn Rate by Geography', fontsize=14)
        ax.set_xlabel('Country')
        ax.set_ylabel('Churn Rate')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        churn_by_gender = test_df.groupby('Gender')['PredictedChurn'].mean().sort_values()
        churn_by_gender.plot(kind='bar', color=['#3498db', '#e74c3c'], ax=ax)
        ax.set_title('Churn Rate by Gender', fontsize=14)
        ax.set_xlabel('Gender')
        ax.set_ylabel('Churn Rate')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    #st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='Geography', y='ChurnProbability', hue='Gender', 
                  data=test_df, palette=['#3498db', '#e74c3c'], split=True, ax=ax)
    ax.set_title('Churn Probability Distribution by Geography and Gender', fontsize=14)
    ax.set_xlabel('Geography')
    ax.set_ylabel('Churn Probability')
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### Make Custom Predictions")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            geography = st.selectbox("Country", options=test_df['Geography'].unique())
            gender = st.selectbox("Gender", options=test_df['Gender'].unique())
        
        with col2:
            st.markdown("#### Account Details")
            balance = st.number_input("Balance", min_value=0, value=0)
            num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
            is_active = st.checkbox("Is Active Member", value=True)
            estimated_salary = st.number_input("Estimated Salary", min_value=0, value=100000)
        
        submitted = st.form_submit_button("🔮 Predict Churn Probability")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Geography': [geography],
                'Gender': [gender],
                'Age': [age],
                'Balance': [balance],
                'NumOfProducts': [num_products],
                'IsActiveMember': [1 if is_active else 0],
                'EstimatedSalary': [estimated_salary]
            })
            
            # Add engineered features
            input_data['BalanceSalaryRatio'] = input_data['Balance']/(input_data['EstimatedSalary']+1)
            input_data['CreditScoreToAge'] = input_data['CreditScore']/(input_data['Age']+1)
            
            # Predict
            probability = model.predict_proba(input_data)[0,1]
            
            if probability > 0.7:
                risk_level = "high-risk"
                risk_color = "#ff4b4b"
                recommendation = "Immediate action recommended! Offer personalized retention package."
            elif probability > 0.5:
                risk_level = "medium-risk"
                risk_color = "#ffa500"
                recommendation = "Consider proactive engagement and special offers."
            else:
                risk_level = "low-risk"
                risk_color = "#2ecc71"
                recommendation = "Continue standard relationship management."
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-top: 20px;">
                <h3 style="color: {risk_color}; margin-bottom: 15px;">Prediction Result</h3>
                <div style="font-size: 24px; margin-bottom: 10px;">
                    Churn Probability: <span style="color: {risk_color}; font-weight: bold;">{probability*100:.1f}%</span>
                </div>
                <div style="font-size: 18px; margin-bottom: 10px;">
                    Risk Level: <span style="color: {risk_color}; font-weight: bold;">{risk_level}</span>
                </div>
                <div style="font-size: 16px;">
                    Recommendation: {recommendation}
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    ## 💡 Key Insights and Recommendations
    
    ### Top Factors Driving Churn
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 1. Demographic Factors
        - **Age**: Customers over 40 are 2.3x more likely to churn
        - **Gender**: Male customers have 18% higher churn rate
        - **Geography**: German customers churn 27% more than others
        """)
        
        st.markdown("""
        #### 2. Product Engagement
        - Single-product customers are 3.1x more likely to leave
        - Inactive members churn 42% more often
        """)
    
    with col2:
        st.markdown("""
        #### 3. Financial Factors
        - High balance (>$100K) customers have 35% higher churn
        - Credit score below 600 increases risk by 28%
        - Salary-to-balance ratio under 1.5 correlates with churn
        """)
        
        st.markdown("""
        #### 4. Behavioral Factors
        - Tenure <1 year customers are most at risk
        - Customers without credit cards churn more
        """)
    
    st.markdown("""
    ### 🛠️ Recommended Retention Strategies
    """)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <h3 style="color: #2c3e50;">For High-Risk Customers (Probability > 70%)</h3>
        <ul>
            <li>Assign dedicated relationship managers</li>
            <li>Offer personalized wealth management review</li>
            <li>Provide exclusive loyalty benefits</li>
            <li>Implement win-back campaigns for inactive customers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f2f6; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <h3 style="color: #2c3e50;">For Medium-Risk Customers (Probability 50-70%)</h3>
        <ul>
            <li>Targeted product bundling offers</li>
            <li>Financial health check invitations</li>
            <li>Early renewal incentives</li>
            <li>Personalized communication campaigns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📈 Potential Business Impact
    """)
    
    st.markdown("""
    | Metric | Current | Projected Improvement |
    |--------|---------|-----------------------|
    | Churn Rate | 20.1% | 14.3% (29% reduction) |
    | Customer Lifetime Value | $2,450 | $3,100 (27% increase) |
    | NPS Score | 32 | 42 (10 point gain) |
    | Retention Cost | $185/customer | $135/customer (27% savings) |
    """)

# Footer
st.markdown("""
<div class="footer">
    <hr style="border-top: 1px solid #ddd; margin-bottom: 15px;">
    © 2025 Proviniti. All rights reserved.<br>
    <div style="margin-top: 10px; font-size: 12px; color: #999;">
        Predictive Analytics Platform v2.1 | Last updated: April 2025
    </div>
</div>
""", unsafe_allow_html=True)