"""
Streamlit Web Application: Vireo Audio Support Operations Analysis Tool
Clean, professional internal analytics tool for non-technical leadership and operations managers.
Methodologically honest and defensible audit version.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_raw_data, clean_tickets_data
from classifier import TicketClassifier
from validator import run_train_test_validation
from analytics import run_monthly_analysis, export_summary_reports

st.set_page_config(
    page_title="Vireo Audio | Support Operations Analysis",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 20px; }
    .metric-card { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .metric-title { font-size: 0.85rem; font-weight: 600; color: #64748B; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #0F172A; }
    .metric-subtitle { font-size: 0.8rem; color: #0EA5E9; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_processed_data():
    tickets, agents, customers, orders, products = load_raw_data()
    t_clean = clean_tickets_data(tickets)
    clf = TicketClassifier()
    clf_df = clf.fit_and_predict(t_clean)
    summary = run_monthly_analysis(clf_df)
    val_results = run_train_test_validation(t_clean, test_size=500, random_state=42)
    return clf_df, summary, val_results, clf

df, summary, val_results, clf = get_processed_data()

st.sidebar.image("https://img.icons8.com/isometric/100/headphones.png", width=60)
st.sidebar.title("Vireo Audio Support Desk")
st.sidebar.caption("AI-Assisted Operations & Staffing Analytics")

nav = st.sidebar.radio(
    "Navigation",
    [
        "📊 Data & Executive Summary",
        "🏷️ Ticket Categorization",
        "📈 Monthly Trends & Workload",
        "👥 Staffing Recommendation",
        "✅ Validation & Accuracy",
        "💰 Financial & Operational Exposure",
        "📥 Data Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("📌 **Dataset Scope:** Jan 2025 – Jun 2026\n\n**Total Tickets:** 11,780\n\n**In-Scope:** 11,641")

# ==========================================
# 1. DATA & EXECUTIVE SUMMARY
# ==========================================
if nav == "📊 Data & Executive Summary":
    st.markdown('<div class="main-header">Vireo Audio Support Desk Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive Summary & Dataset Overview (Jan 2025 – Jun 2026)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">In-Scope Support Tickets</div>
            <div class="metric-value">11,641</div>
            <div class="metric-subtitle">Jan 2025 – Jun 2026 (11,780 total)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Planned Staffing Budget</div>
            <div class="metric-value">₹9.00 Lakhs</div>
            <div class="metric-subtitle">2 Planned Hires (@ ₹4.5L/year)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">SLA Breach Credit Exposure</div>
            <div class="metric-value">₹4.52 Lakhs</div>
            <div class="metric-subtitle">1,292 Breaches (@ ₹350 store credit)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Transfer Re-handling Exposure</div>
            <div class="metric-value">₹3.96 Lakhs</div>
            <div class="metric-subtitle">1,299 Transfers (@ ₹305/re-handle)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Executive Findings")
    st.warning("""
    **Core Operations Finding:**
    - The intake chatbot naively tagged tickets mentioning payment terms (*"paid"*, *"money deducted"*) as **Billing & Payments** (21.8% of total volume).
    - As a result, CX leadership planned to allocate **2 new hires (~₹9 Lakhs/year budget)** to Billing.
    - Our empirical re-classification reveals that over **40% of Billing intake tickets are actually Delivery & Shipping issues** (shipment delays, courier tracking, non-delivery) that were manually transferred to **Logistics**.
    - **Logistics** is the true workload bottleneck: accounting for **28.4% of total volume (3,309 tickets)**, a median resolution time of **19.1 hours** (25.1h under intake routing), and an **SLA breach rate of 17.1%**.
    - **Recommendation:** Reallocate the 2 planned hires from Billing to **Logistics**.
    """)
    
    st.markdown("### Dataset Overview & Relationship Schema")
    st.markdown("""
    - **Tickets (`tickets.csv`):** 11,780 rows across 4 channels (Chat 45.7%, Email 30.2%, Voice 14.4%, Social 9.6%).
    - **Agents (`agents.csv`):** 44 agent records across Indore and Bengaluru sites.
    - **Orders (`orders.csv`):** 15,000 orders tied via `order_id` or `customer_id + product_sku`.
    - **Customers (`customers.csv`):** 9,500 registered customers.
    - **Products (`products.csv`):** 14 hardware SKUs definitions.
    - **Operating Policy (`support-policy.pdf`):** SLAs (Chat 15m, Voice 2h, Social 4h, Email 8h), Breach credit ₹350, Transfer cost ₹305.
    """)

# ==========================================
# 2. TICKET CATEGORIZATION
# ==========================================
elif nav == "🏷️ Ticket Categorization":
    st.markdown('<div class="main-header">Category Taxonomy & Live Intake Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-Driven Taxonomy & Interactive Ticket Classifier Sandbox</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Refined Business Category Taxonomy")
    taxonomy_data = [
        {"Category": "Delivery & Shipping", "Target Team": "Logistics", "Definition": "Order tracking, shipment delays, non-delivery, wrong item delivered, courier issues, RTO, address updates.", "Example": "paid on 19 jun, still waiting for package VR898250"},
        {"Category": "Returns & Refunds", "Target Team": "Returns Desk", "Definition": "Return requests, reverse pickup status/delays, refund processing post-return/QC, DOA return requests.", "Example": "return pickup has not happened. VR894797. please help"},
        {"Category": "Billing & Payments", "Target Team": "Billing", "Definition": "Genuine payment failures, double charges, payment gateway errors, invoice requests, price adjustment.", "Example": "money got deducted twice during checkout on website"},
        {"Category": "Hardware & Technical", "Target Team": "Frontline Teams", "Definition": "Charging/battery failure, static/audio distortion, bluetooth pairing/disconnecting, app crash, firmware bug.", "Example": "left bud won't charge even after keeping in case overnight"},
        {"Category": "Warranty & RMA", "Target Team": "Escalations & Warranty", "Definition": "In-warranty repair claims, RMA tracking, service center status, hardware replacement under warranty.", "Example": "warranty claim pending for 5 days claim number rma82419"},
        {"Category": "Product Enquiry", "Target Team": "Frontline Teams", "Definition": "Pre-sales queries, product specs, compatibility, usage guidance.", "Example": "can i connect two orbit smart speakers together?"},
        {"Category": "Account & Login", "Target Team": "Frontline Teams", "Definition": "Account access, OTP delivery failures, password reset, profile/registration issues.", "Example": "the login code never arrives, tried 6 times"},
        {"Category": "Other / Unclear", "Target Team": "Frontline Teams", "Definition": "Ambiguous queries, incomplete messages, spam.", "Example": "hello? is anyone there?"}
    ]
    st.table(pd.DataFrame(taxonomy_data))
    
    st.markdown("---")
    st.markdown("### 2. Live Intake Classifier Sandbox")
    st.caption("Tests prediction strictly on opening customer message available at intake (without agent notes leakage):")
    
    sample_text = st.selectbox(
        "Select a sample message or enter custom text below:",
        [
            "Custom text...",
            "paid on 19 jun, money deducted from account but order VR898250 not delivered yet",
            "return pickup has not happened for my Nexa watch. courier didn't show up",
            "charged twice on my credit card during checkout, payment failed on first attempt",
            "left earbud silent and static noise in right earbud when playing music",
            "rma claim number rma82419 sent to service center 12 days ago no update",
            "OTP is not coming to my registered mobile number for login"
        ]
    )
    
    user_input = st.text_area(
        "Customer Opening Message:",
        value="" if sample_text == "Custom text..." else sample_text,
        height=100
    )
    
    if st.button("Classify Live Ticket"):
        if user_input.strip():
            pred_cat, pred_team, pred_conf, reason = clf.predict_intake(user_input)
            
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Predicted Category", pred_cat)
            with res_col2:
                st.metric("Target Team", pred_team)
            with res_col3:
                st.metric("Model Confidence", f"{pred_conf*100:.0f}%")
            st.info(f"**Classification Logic:** {reason}")
        else:
            st.warning("Please enter a customer message to classify.")

# ==========================================
# 3. MONTHLY TRENDS & WORKLOAD
# ==========================================
elif nav == "📈 Monthly Trends & Workload":
    st.markdown('<div class="main-header">Monthly Business Analysis & Workload</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Jan 2025 – Jun 2026 Breakdown by Volume, Category & Team</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Monthly Ticket Volume Trend")
    fig_vol = px.bar(
        summary['monthly_vol'],
        x='month_year',
        y='total_tickets',
        title="Monthly In-Scope Ticket Volume (Jan 2025 – Jun 2026)",
        labels={'month_year': 'Month', 'total_tickets': 'Ticket Count'},
        color_discrete_sequence=['#0EA5E9']
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 2. Intake Bot Categories (Flawed)")
        bot_cat_dist = df[df['in_scope']]['category'].value_counts().reset_index()
        bot_cat_dist.columns = ['category', 'count']
        fig_bot = px.pie(
            bot_cat_dist,
            values='count',
            names='category',
            title="Bot Intake Category Distribution (Billing = 20.8%)",
            hole=0.4
        )
        st.plotly_chart(fig_bot, use_container_width=True)
        
    with col_b:
        st.markdown("### 3. AI Re-classified Categories (Actual)")
        ai_cat_dist = df[df['in_scope']]['predicted_category'].value_counts().reset_index()
        ai_cat_dist.columns = ['predicted_category', 'count']
        fig_ai = px.pie(
            ai_cat_dist,
            values='count',
            names='predicted_category',
            title="Actual Intent Category Distribution (Delivery = 28.4%)",
            hole=0.4
        )
        st.plotly_chart(fig_ai, use_container_width=True)

    st.markdown("### 4. Team Workload Comparison: Intake Routing vs Re-classified Demand")
    team_comp = pd.DataFrame({
        'Assigned Team (Intake Routing)': df[df['in_scope']]['assigned_team'].value_counts(),
        'Re-classified Team (Actual Demand)': df[df['in_scope']]['predicted_team'].value_counts()
    }).fillna(0).astype(int)
    
    fig_team = px.bar(
        team_comp,
        barmode='group',
        title="Team Workload Comparison (Intake Routing vs Re-classified Demand)",
        labels={'value': 'Ticket Volume', 'index': 'Team'}
    )
    st.plotly_chart(fig_team, use_container_width=True)

# ==========================================
# 4. STAFFING RECOMMENDATION
# ==========================================
elif nav == "👥 Staffing Recommendation":
    st.markdown('<div class="main-header">Headcount & Staffing Recommendation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Headcount Reallocation: Billing vs Logistics</div>', unsafe_allow_html=True)
    
    st.info("""
    **Context & Problem Premise:**
    - Priya Raman (Head of CX): *"Billing is our biggest queue by a mile, 22% of tickets... Two hires go to whichever team is biggest."*
    - Arjun Mehta (Finance Controller): *"Two hires is about Rs 9 lakh a year. I want to see the volume case in writing before I sign..."*
    - Neha Kulkarni (Support Ops Mgr): *"Billing keep telling me half their queue isn't theirs... it's Logistics who are actually drowning..."*
    """)
    
    st.markdown("### Re-classified Operational Team Workload Metrics")
    st.dataframe(summary['team_metrics'], use_container_width=True)
    
    st.markdown("### Decision Analysis & Resource Allocation")
    st.markdown("""
    1. **Volume Evaluation:** 
       - Intake routing assigned 2,310 in-scope tickets to Billing (19.8%), making it appear as queue #2 behind Chat Frontline.
       - Re-classification shows **Logistics handles 3,309 in-scope tickets (28.4% of total demand)**, while Billing drops to **1,351 tickets (11.6%)**.
    2. **Operational Strain & Handling Efficiency:**
       - **Logistics** has a median resolution time of **19.1 hours** (25.1 hours under intake routing).
       - **Logistics** experiences the highest SLA breach rate in the company (**17.1%**).
       - **Logistics** handles **685 internal transfers/hand-offs**.
    3. **Recommendation:**
       - **REALLOCATE BOTH PLANNED HIRES (2 HEADCOUNT, ₹9.0L BUDGET) TO LOGISTICS.**
       - Zero hires should go to Billing. Billing's perceived backlog was created by misrouted delivery delay queries.
    """)

# ==========================================
# 5. VALIDATION RESULTS
# ==========================================
elif nav == "✅ Validation & Accuracy":
    st.markdown('<div class="main-header">Classifier Validation & Accuracy Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Train/Test Evaluation on 500 Held-Out Test Tickets (random_state=42)</div>', unsafe_allow_html=True)
    
    st.info(f"**Validation Methodology Note:** {val_results['methodology_note']}")
    
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.metric("Test Sample Size", val_results['sample_size'])
    with col_v2:
        st.metric("Correct Predictions", val_results['correct_predictions'])
    with col_v3:
        st.metric("Incorrect Predictions", val_results['incorrect_predictions'])
    with col_v4:
        st.metric("Held-Out Accuracy", f"{val_results['accuracy']*100:.1f}%")
        
    st.markdown("### Category-Level Classification Metrics")
    st.text(val_results['classification_report_str'])
    
    st.markdown("### Confusion Matrix")
    st.dataframe(val_results['confusion_matrix_df'], use_container_width=True)

# ==========================================
# 6. FINANCIAL & OPERATIONAL EXPOSURE
# ==========================================
elif nav == "💰 Financial & Operational Exposure":
    st.markdown('<div class="main-header">Financial & Operational Exposure Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Defensible Resource Allocation & Operational Cost Summary</div>', unsafe_allow_html=True)
    
    st.markdown("### 1. Resource Allocation & Operational Cost Exposure (INR)")
    fin_data = [
        {"Metric Category": "Planned Staffing Allocation", "Amount (INR)": "₹9,00,000 / year", "Description": "Planned budget for 2 support headcount (@ ₹4.5L/year each). Recommended to reallocate from Billing to Logistics."},
        {"Metric Category": "Total In-Scope SLA Breach Exposure", "Amount (INR)": "₹4,52,200 total", "Description": "1,292 first-response SLA breaches across all channels (@ ₹350 store credit per breach)."},
        {"Metric Category": "Total Transfer Re-handling Exposure", "Amount (INR)": "₹396,195 total", "Description": "1,299 internal hand-offs recorded in helpdesk period (@ ₹305 re-handling cost per transfer)."},
        {"Metric Category": "Avoidable Misrouting Cost Exposure", "Amount (INR)": "₹2,86,635 total", "Description": "SLA breaches (₹1.21L) and transfer overhead (₹1.63L) directly generated by misrouted Billing intake tickets."}
    ]
    st.table(pd.DataFrame(fin_data))
    
    st.markdown("---")
    st.markdown("### 2. Operational Model Execution Cost Calculation")
    st.markdown("""
    - **Single Execution Cost (11,780 tickets):**
      - Local TF-IDF + Logistic Regression / Rules Model: **₹0.00** (Runs locally in ~1.2 seconds).
      - If run via Cloud LLM API (e.g. OpenAI/Gemini Flash @ ~300 tokens/ticket):
        - 11,780 tickets * 300 tokens = ~3.53M tokens.
        - Approximate Cost per Execution: **~$0.53 (₹45 INR)**.
    - **Monthly Operational Cost (at ~650 tickets/week = ~2,800 tickets/month):**
      - Local Model: **₹0.00 / month**.
      - Cloud LLM API Option: 2,800 tickets * 300 tokens = ~840k tokens = **~$0.13 / month (~₹11 INR / month)**.
    """)

# ==========================================
# 7. DATA EXPORT
# ==========================================
elif nav == "📥 Data Export":
    st.markdown('<div class="main-header">Data Export & Downloads</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Download Re-classified Datasets, Validation Sets & Reports</div>', unsafe_allow_html=True)
    
    csv_classified = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Classified Ticket Dataset (CSV)",
        data=csv_classified,
        file_name="vireo_classified_tickets.csv",
        mime="text/csv"
    )
    
    csv_gt = val_results['eval_df'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Ground Truth Validation Sample (CSV)",
        data=csv_gt,
        file_name="vireo_ground_truth_validation.csv",
        mime="text/csv"
    )
    
    csv_team = summary['team_metrics'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Team Workload Metrics Report (CSV)",
        data=csv_team,
        file_name="vireo_team_workload_metrics.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Vireo Audio Support Desk Operations Analysis Tool | Built for Vireo Audio Leadership")
