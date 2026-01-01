import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np

# === KONFIGURASI STREAMLIT ===
st.set_page_config(
    page_title="Demographics & Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",  # Biar lebar penuh seperti dashboard bank
    initial_sidebar_state="expanded"
)

# Judul besar di atas
st.title("📊 Demographics & Churn Prediction Dashboard")
st.markdown("_Evaluating Current Clients | Churn Risk Analysis (Proxy Model)_")

# === LOAD DATA ===
@st.cache_data  # Agar data hanya dimuat sekali
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

# === HITUNG METRICS ===
total_clients = len(df)
male_count = df[df['Gender'] == 'Male'].shape[0]
female_count = df[df['Gender'] == 'Female'].shape[0]
avg_income = df['TotalPrice'].mean()
avg_recency = df['Recency'].mean()
predicted_high_risk = df['Predicted_Churn'].sum()
predicted_churn_rate = (predicted_high_risk / total_clients) * 100

# === BUAT DASHBOARD MATPLOTLIB ===
fig = plt.figure(figsize=(30, 22), facecolor='#f0f4f8')
gs = GridSpec(7, 5, figure=fig, hspace=0.7, wspace=0.6)

pink = '#FF6B9D'
navy = '#2C3E50'
green = '#27AE60'
red = '#E74C3C'
pastel_colors = sns.color_palette("pastel", 5)
set2_colors = sns.color_palette("Set2", 5)

fig.patch.set_edgecolor('#BDC3C7')
fig.patch.set_linewidth(2)

# (Sisanya SAMA PERSIS dengan kode Matplotlib Anda sebelumnya)
# 1. Total Clients
ax_total = fig.add_subplot(gs[0:2, 0])
ax_total.axis('off')
ax_total.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, facecolor='white', alpha=0.9, edgecolor=navy, linewidth=2))
ax_total.text(0.5, 0.7, f"{total_clients:,}", ha='center', va='center', fontsize=48, fontweight='bold', color=navy)
ax_total.text(0.5, 0.3, 'Total Clients', ha='center', va='center', fontsize=16, color='#555')

# 2. Gender
ax_gender = fig.add_subplot(gs[0:2, 1])
bars = ax_gender.bar(['Female', 'Male'], [female_count, male_count], color=[pink, navy], width=0.7)
ax_gender.set_title('Gender Breakdown', fontsize=18, fontweight='bold', color=navy)
ax_gender.set_ylim(0, max(male_count, female_count) * 1.3)
for bar in bars:
    h = bar.get_height()
    ax_gender.text(bar.get_x() + bar.get_width()/2, h + 20, f'{int(h)}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# 3. Avg Income
ax_income = fig.add_subplot(gs[0:2, 2])
ax_income.axis('off')
ax_income.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, facecolor='white', alpha=0.9, edgecolor=pink, linewidth=2))
ax_income.text(0.5, 0.7, f"${avg_income:,.0f}", ha='center', va='center', fontsize=36, fontweight='bold', color=pink)
ax_income.text(0.5, 0.3, 'Avg Yearly Income', ha='center', va='center', fontsize=14, color='#555')

# 4. Avg Recency
ax_age = fig.add_subplot(gs[0:2, 3])
ax_age.axis('off')
ax_age.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, facecolor='white', alpha=0.9, edgecolor=navy, linewidth=2))
ax_age.text(0.5, 0.7, f"{avg_recency:.0f} days", ha='center', va='center', fontsize=48, fontweight='bold', color=navy)
ax_age.text(0.5, 0.3, 'Avg Recency\n(Age Proxy)', ha='center', va='center', fontsize=14, color='#555')

# 5. Predicted Churn Rate
ax_churn = fig.add_subplot(gs[0:2, 4])
ax_churn.axis('off')
ax_churn.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, facecolor='white', alpha=0.9, edgecolor=red if predicted_churn_rate > 30 else green, linewidth=3))
ax_churn.text(0.5, 0.7, f"{predicted_churn_rate:.1f}%", ha='center', va='center', fontsize=48, fontweight='bold', color=red if predicted_churn_rate > 30 else green)
ax_churn.text(0.5, 0.3, 'Predicted\nChurn Rate', ha='center', va='center', fontsize=16, color='#555', fontweight='bold')

# 6. Age Group Bar Chart (FIXED)
ax_age_bar = fig.add_subplot(gs[2:4, 2:5])
age_gender = pd.crosstab(df['Age_Group'], df['Gender'])
# Pastikan urutan Female → Male
if 'Female' in age_gender.columns and 'Male' in age_gender.columns:
    age_gender = age_gender[['Female', 'Male']]
elif 'Female' in age_gender.columns:
    age_gender = age_gender[['Female']]
elif 'Male' in age_gender.columns:
    age_gender = age_gender[['Male']]

age_gender.plot(kind='bar', ax=ax_age_bar, color=[pink, navy][:age_gender.shape[1]], width=0.8, edgecolor='white')
ax_age_bar.set_title('Clients by Age Group & Gender', fontsize=20, fontweight='bold', color=navy)
ax_age_bar.set_ylabel('Number of Clients')
ax_age_bar.grid(axis='y', linestyle='--', alpha=0.5)
for c in ax_age_bar.containers:
    ax_age_bar.bar_label(c, fmt='%d', fontsize=12, padding=3)

# 7. Income Pie
ax_income_pie = fig.add_subplot(gs[2, 0:2])
income_vals = df['Income_Group'].value_counts()
explode = [0.1 if i == income_vals.argmax() else 0.05 for i in range(len(income_vals))]
ax_income_pie.pie(income_vals, labels=income_vals.index,
                  autopct=lambda p: f'{p:.1f}%\n({int(p/100*len(df))})',
                  colors=pastel_colors, explode=explode, shadow=True, textprops={'fontsize': 12})
ax_income_pie.set_title('Income Group Distribution', fontsize=18, fontweight='bold', color=navy)

# 8. Credit Pie
ax_credit_pie = fig.add_subplot(gs[3, 0:2])
credit_vals = df['Credit_Score'].value_counts()
explode = [0.1 if i == credit_vals.argmax() else 0.05 for i in range(len(credit_vals))]
ax_credit_pie.pie(credit_vals, labels=credit_vals.index,
                  autopct=lambda p: f'{p:.1f}%\n({int(p/100*len(df))})',
                  colors=set2_colors, explode=explode, shadow=True, textprops={'fontsize': 12})
ax_credit_pie.set_title('Credit Score Distribution', fontsize=18, fontweight='bold', color=navy)

# 9. Churn Risk by Age Group
ax_churn_age = fig.add_subplot(gs[4:6, 0:2])
churn_by_age = df.groupby('Age_Group')['Predicted_Churn'].mean() * 100
churn_by_age.plot(kind='bar', ax=ax_churn_age, color=[red if x > 50 else '#FF9F40' for x in churn_by_age], width=0.7)
ax_churn_age.set_title('Churn Risk % by Age Group', fontsize=18, fontweight='bold', color=navy)
ax_churn_age.set_ylabel('Churn Risk (%)')
ax_churn_age.set_ylim(0, 100)
for i, v in enumerate(churn_by_age):
    ax_churn_age.text(i, v + 3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)

# 10. Churn Risk by Income Group
ax_churn_income = fig.add_subplot(gs[4:6, 2:4])
churn_by_income = df.groupby('Income_Group')['Predicted_Churn'].mean() * 100
churn_by_income.plot(kind='bar', ax=ax_churn_income, color=[red if x > 50 else '#FF9F40' for x in churn_by_income], width=0.7)
ax_churn_income.set_title('Churn Risk % by Income Group', fontsize=18, fontweight='bold', color=navy)
ax_churn_income.set_ylabel('Churn Risk (%)')
ax_churn_income.set_ylim(0, 100)
for i, v in enumerate(churn_by_income):
    ax_churn_income.text(i, v + 3, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)

# 11. Risk Category Summary
ax_risk = fig.add_subplot(gs[4:6, 4])
risk_count = df['Risk_Category'].value_counts()
bars = ax_risk.bar(risk_count.index, risk_count.values, color=[green, red], width=0.6)
ax_risk.set_title('Total by Risk Category', fontsize=18, fontweight='bold', color=navy)
for bar in bars:
    h = bar.get_height()
    ax_risk.text(bar.get_x() + bar.get_width()/2, h + 10, f'{int(h)}', ha='center', fontsize=16, fontweight='bold')

# Recommendations
ax_recom = fig.add_subplot(gs[6, 0:5])
ax_recom.axis('off')
highest_age_risk = churn_by_age.idxmax()
recommendations = f"""
🔴 CHURN PREDICTION INSIGHTS & RECOMMENDATIONS

• Predicted Churn Rate: {predicted_churn_rate:.1f}% ({predicted_high_risk:,} high-risk customers)
• Highest Churn Risk Age Group: {highest_age_risk} ({churn_by_age.max():.1f}% risk)
• Key Driver: Long recency (>180 days) strongly linked to churn

📈 Recommended Actions:
→ Re-engagement campaign for inactive >90 days
→ Personalized offers for Low & Lower Middle income segments
→ Loyalty rewards for 41-70 age group (high volume + risk)
→ Monitor low-frequency customers closely

Goal: Reduce churn by 15-20% in next quarter
"""
ax_recom.text(0.02, 0.95, recommendations, fontsize=15, va='top', ha='left',
              bbox=dict(boxstyle="round,pad=1", facecolor='white', edgecolor=navy, linewidth=2),
              color='#2c3e50', linespacing=1.6)

# Contoh bagian akhir (sama seperti sebelumnya)
ax_recom = fig.add_subplot(gs[6, 0:5])
ax_recom.axis('off')
highest_age_risk = df.groupby('Age_Group')['Predicted_Churn'].mean().idxmax()
recommendations = f"""
🔴 CHURN PREDICTION INSIGHTS & RECOMMENDATIONS

• Predicted Churn Rate: {predicted_churn_rate:.1f}% ({predicted_high_risk:,} high-risk customers)
• Highest Churn Risk Age Group: {highest_age_risk}
• Key Driver: Long recency (>180 days) strongly linked to churn

📈 Recommended Actions:
→ Re-engagement campaign for inactive >90 days
→ Personalized offers for Low & Lower Middle income segments
→ Loyalty rewards for 41-70 age group
→ Monitor low-frequency customers closely

Goal: Reduce churn by 15-20% in next quarter
"""
ax_recom.text(0.02, 0.95, recommendations, fontsize=15, va='top', ha='left',
              bbox=dict(boxstyle="round,pad=1", facecolor='white', edgecolor=navy, linewidth=2),
              color='#2c3e50', linespacing=1.6)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# === TAMPILKAN DI STREAMLIT ===
st.pyplot(fig, use_container_width=True)

# Tambahan: Info di sidebar
with st.sidebar:
    st.header("📌 Informasi")
    st.write(f"**Total Data:** {total_clients:,} pelanggan")
    st.write(f"**Predicted Churn Rate:** {predicted_churn_rate:.1f}%")
    st.write("Data source: `churn_results.csv`")
    st.caption("Dashboard by Streamlit + Matplotlib")
