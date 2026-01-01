import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === KONFIGURASI STREAMLIT ===
st.set_page_config(
    page_title="Demographics & Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk UI/UX lebih menarik
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stButton > button { background-color: #2C3E50; color: white; border-radius: 5px; }
    .stMetric { background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .sidebar .sidebar-content { background-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv('churn_results.csv', sep=';')
    df.columns = df.columns.str.strip()
    
    for col in ['TotalPrice', 'UnitPrice', 'Quantity', 'Recency']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['CustomerID'] = df['CustomerID'].astype(int)
    df = df.dropna(subset=['Recency'])
    
    # Proxy variabel
    df['Gender'] = df['Churn'].map({1: 'Male', 0: 'Female'})
    
    max_recency = df['Recency'].max()
    age_bins = [0, 60, 120, 180, 240, 300, max_recency + 1]
    age_labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '>71']
    df['Age_Group'] = pd.cut(df['Recency'], bins=age_bins, labels=age_labels, include_lowest=True)
    df = df.dropna(subset=['Age_Group'])
    
    df['Income_Group'] = pd.cut(df['TotalPrice'], bins=5,
                                labels=['Low Income', 'Lower Middle', 'Middle', 'Upper Middle', 'High Income'])
    
    df['Credit_Score'] = pd.cut(df['Quantity'], bins=5,
                                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    df['Risk_Category'] = df['Predicted_Churn'].map({1: 'High Risk', 0: 'Low Risk'})
    
    return df

df = load_data()

# === SIDEBAR: FILTERS & INFO (INTERAKTIF) ===
with st.sidebar:
    st.header("🔍 Filters")
    
    # Filter by Country (asli dari data, proxy angka → nama jika ada mapping)
    countries = sorted(df['Country'].unique())
    selected_country = st.multiselect("Select Country", countries, default=countries)
    
    # Filter by Age Group
    age_groups = sorted(df['Age_Group'].unique())
    selected_age = st.multiselect("Select Age Group", age_groups, default=age_groups)
    
    # Filter by Risk Category
    risks = df['Risk_Category'].unique()
    selected_risk = st.multiselect("Select Risk Category", risks, default=risks)
    
    # Apply filters
    filtered_df = df[
        (df['Country'].isin(selected_country)) &
        (df['Age_Group'].isin(selected_age)) &
        (df['Risk_Category'].isin(selected_risk))
    ]
    
    st.divider()
    st.header("📌 Quick Info")
    st.write(f"**Filtered Clients:** {len(filtered_df):,}")
    st.write(f"**Avg Income:** ${filtered_df['TotalPrice'].mean():,.0f}")
    st.write(f"**Avg Recency:** {filtered_df['Recency'].mean():.0f} days")
    
    if st.button("Reset Filters"):
        st.experimental_rerun()

# === MAIN CONTENT ===
# Key Metrics di atas (lebih informatif dengan st.metric)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Clients", f"{len(filtered_df):,}", help="Jumlah pelanggan setelah filter")
with col2:
    st.metric("Male / Female", f"{filtered_df[filtered_df['Gender'] == 'Male'].shape[0]:,} / {filtered_df[filtered_df['Gender'] == 'Female'].shape[0]:,}", help="Rasio gender")
with col3:
    st.metric("Avg Income", f"${filtered_df['TotalPrice'].mean():,.0f}", help="Rata-rata belanja tahunan")
with col4:
    st.metric("Avg Recency", f"{filtered_df['Recency'].mean():.0f} days", help="Rata-rata hari sejak transaksi terakhir")
with col5:
    churn_rate = (filtered_df['Predicted_Churn'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    delta_color = "inverse" if churn_rate > 30 else "normal"
    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%", help="Persentase prediksi churn", delta_color=delta_color)

# Tabs untuk organisasi konten (lebih UX friendly)
tab1, tab2, tab3 = st.tabs(["📈 Demographics", "🚨 Churn Analysis", "📋 High-Risk List"])

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Gender Breakdown (interaktif dengan Plotly)
        gender_count = filtered_df['Gender'].value_counts().reset_index()
        fig_gender = px.bar(gender_count, x='Gender', y='count', color='Gender',
                            color_discrete_map={'Male': '#2C3E50', 'Female': '#FF6B9D'},
                            title="Gender Breakdown",
                            labels={'count': 'Number of Clients'})
        fig_gender.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_gender, use_container_width=True)
        
        # Income Pie
        income_vals = filtered_df['Income_Group'].value_counts().reset_index()
        fig_income = px.pie(income_vals, values='count', names='Income_Group',
                            title="Income Group Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel,
                            hole=0.3)  # Donut style untuk menarik
        fig_income.update_traces(textposition='inside', textinfo='percent+label')
        fig_income.update_layout(height=400)
        st.plotly_chart(fig_income, use_container_width=True)
    
    with col_right:
        # Age Group Bar (grouped by Gender)
        age_gender = pd.crosstab(filtered_df['Age_Group'], filtered_df['Gender']).reset_index()
        fig_age = px.bar(age_gender.melt(id_vars='Age_Group'), x='Age_Group', y='value', color='Gender',
                         color_discrete_map={'Male': '#2C3E50', 'Female': '#FF6B9D'},
                         title="Clients by Age Group & Gender",
                         barmode='group',
                         labels={'value': 'Number of Clients'})
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Credit Score Pie
        credit_vals = filtered_df['Credit_Score'].value_counts().reset_index()
        fig_credit = px.pie(credit_vals, values='count', names='Credit_Score',
                            title="Credit Score Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            hole=0.3)
        fig_credit.update_traces(textposition='inside', textinfo='percent+label')
        fig_credit.update_layout(height=400)
        st.plotly_chart(fig_credit, use_container_width=True)

with tab2:
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Churn Risk by Age Group
        churn_by_age = (filtered_df.groupby('Age_Group')['Predicted_Churn'].mean() * 100).reset_index()
        fig_churn_age = px.bar(churn_by_age, x='Age_Group', y='Predicted_Churn',
                               title="Churn Risk % by Age Group",
                               color='Predicted_Churn', color_continuous_scale='OrRd',
                               labels={'Predicted_Churn': 'Churn Risk (%)'})
        fig_churn_age.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_churn_age, use_container_width=True)
    
    with col_right:
        # Churn Risk by Income Group
        churn_by_income = (filtered_df.groupby('Income_Group')['Predicted_Churn'].mean() * 100).reset_index()
        fig_churn_income = px.bar(churn_by_income, x='Income_Group', y='Predicted_Churn',
                                  title="Churn Risk % by Income Group",
                                  color='Predicted_Churn', color_continuous_scale='OrRd',
                                  labels={'Predicted_Churn': 'Churn Risk (%)'})
        fig_churn_income.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_churn_income, use_container_width=True)
    
    # Risk Category Distribution (interaktif pie)
    risk_count = filtered_df['Risk_Category'].value_counts().reset_index()
    fig_risk = px.pie(risk_count, values='count', names='Risk_Category',
                      title="Total Clients by Risk Category",
                      color_discrete_map={'Low Risk': '#27AE60', 'High Risk': '#E74C3C'},
                      hole=0.3)
    fig_risk.update_traces(textposition='inside', textinfo='percent+label')
    fig_risk.update_layout(height=400)
    st.plotly_chart(fig_risk, use_container_width=True)

with tab3:
    st.subheader("🚨 High-Risk Customers List")
    st.markdown("Daftar pelanggan dengan prediksi churn tinggi (Predicted_Churn = 1). Klik kolom untuk sort.")
    
    high_risk = filtered_df[filtered_df['Predicted_Churn'] == 1][['CustomerID', 'Country', 'Age_Group', 'TotalPrice', 'Recency', 'Risk_Category']]
    high_risk['TotalPrice'] = high_risk['TotalPrice'].map('${:,.0f}'.format)
    st.dataframe(high_risk, use_container_width=True, height=400)
    
    # Download button untuk CSV
    csv = high_risk.to_csv(index=False)
    st.download_button("📥 Download High-Risk List (CSV)", csv, "high_risk_customers.csv", "text/csv")

# === SUMMARY & RECOMMENDATIONS (INFORMATIF) ===
st.divider()
st.header("🔴 Summary Insights & Recommendations")
col_sum1, col_sum2 = st.columns(2)

with col_sum1:
    st.subheader("Key Insights")
    st.markdown(f"""
    - **Total Filtered Clients:** {len(filtered_df):,}
    - **Predicted Churn Rate:** {predicted_churn_rate:.1f}% ({predicted_high_risk:,} at risk)
    - **Highest Risk Age Group:** {filtered_df.groupby('Age_Group')['Predicted_Churn'].mean().idxmax()}
    - **Average Income:** ${filtered_df['TotalPrice'].mean():,.0f}
    - **Average Recency:** {filtered_df['Recency'].mean():.0f} days
    """)

with col_sum2:
    st.subheader("Recommended Actions")
    st.markdown("""
    - **Re-engagement Campaign:** Target customers inactive >90 days with personalized emails.
    - **Discount Offers:** For Low & Lower Middle income groups to reduce churn.
    - **Loyalty Program:** Reward frequent buyers in 41-70 age group.
    - **Monitoring:** Track low-frequency transactions and send reminders.
    """)
    st.info("Goal: Reduce churn by 15-20% in the next quarter through targeted interventions.")

# Footer
st.markdown("---")
st.caption("Dashboard created with Streamlit | Data as of January 01, 2026 | For demo purposes only")
