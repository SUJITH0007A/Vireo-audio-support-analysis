# Vireo Audio — Support Operations Analysis & AI Ticket Classifier

An AI-assisted support-ticket analysis and re-classification tool built for **Vireo Audio** (Bengaluru, India).  
This tool ingests raw helpdesk exports, re-classifies customer support tickets into a data-driven 8-category taxonomy at intake, identifies true team workload bottlenecks, evaluates prediction accuracy on a held-out test dataset, and provides defensible financial framing for headcount allocation decisions.

---

## 📁 Project Structure

```text
c:\1.My projects\Task 1\
├── given files/                        # Raw, unmodified supplied dataset files
│   ├── tickets.csv.csv                 # 11,780 support tickets (1 Jan 2025 – 30 Jun 2026)
│   ├── agents.csv.csv                  # 44 agent roster records
│   ├── customer.csv.csv                # 9,500 customer records
│   ├── orders.csv.csv                  # 15,000 order records
│   ├── products.csv.csv                # 14 product SKU definitions
│   ├── ReadMe.txt and email-thread.txt # Schema docs and internal email thread
│   └── support policy.pdf              # Support Operating Policy v3.2 (SLAs, costs, rules)
├── src/                                # Core Python modules
│   ├── data_loader.py                  # Ingestion, cleaning, timestamp parsing, SLA & cost calculations
│   ├── classifier.py                  # Hybrid classifier (Live intake mode vs Historical reclassification)
│   ├── validator.py                   # Train/Test evaluation & rule-based ground-truth validation
│   └── analytics.py                   # Monthly trends, team workload, and financial exposure metrics
├── outputs/                            # Generated output reports & datasets
│   ├── classified_tickets.csv          # Re-classified full ticket dataset
│   ├── ground_truth_sample.csv         # 500-ticket held-out evaluation dataset
│   ├── monthly_volume.csv              # Monthly ticket volume trends
│   ├── monthly_category_breakdown.csv  # Monthly volume by category
│   ├── monthly_team_breakdown.csv      # Monthly volume by team
│   └── team_workload_metrics.csv       # Team performance & SLA metrics
├── app.py                              # Interactive Streamlit Web Application
├── cli.py                              # Command-Line Application interface
├── business_memo.md                    # One-page executive memo for Head of CX (Priya Raman)
├── submission-form.md                  # Complete assignment submission responses
├── screen_recording_script.md          # 3-minute video presentation script
├── requirements.txt                    # Python dependencies
└── .env.example                        # Template for optional API environment variables
```

---

## 🛠️ Prerequisites & Setup

1. **Python Environment:** Python 3.10+ (Python 3.14 compatible).
2. **Install Dependencies:**
   ```powershell
   cd "c:\1.My projects\Task 1"
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run

### Option 1: Run Interactive Streamlit Web Dashboard
```powershell
streamlit run app.py
```
Open browser at `http://localhost:8501`.

### Option 2: Run Command Line Application (CLI)
```powershell
python cli.py
```

---

## 🤖 Model Approach & Live vs Historical Separation

### 1. Separation of Classification Modes
- **Live Intake Classification (`predict_intake`):** Predicts ticket category and operational team using **ONLY the opening customer message** (`customer_message`). Does **NOT** use `agent_notes` (which do not exist yet at ticket creation).
- **Retrospective Historical Analysis (`reclassify_historical`):** Evaluates combined customer message and agent closing notes (`agent_notes`) to audit historic intent and generate proxy labels.

### 2. Hybrid Classifier Engine (`src/classifier.py`)
- **Domain Heuristic Layer:** High-precision regex patterns identify unambiguous indicators (RMA numbers, courier tracking/AWBs, return pickups, OTP login errors).
- **TF-IDF + Logistic Regression Layer:** Vectorizes n-grams (1,2) on customer opening text and predicts category with calibrated probabilities.
- **Team Mapping:**
  - `Delivery & Shipping` ➔ **Logistics**
  - `Returns & Refunds` ➔ **Returns Desk**
  - `Billing & Payments` ➔ **Billing**
  - `Warranty & RMA` ➔ **Escalations & Warranty**
  - `Hardware & Technical` / `Product Enquiry` / `Account & Login` / `Other` ➔ **Frontline Teams (Chat/Email/Voice)**

---

## ✅ Validation Methodology

1. **Train/Test Evaluation Split (`src/validator.py`):**
   - The dataset is split into training (11,280 tickets) and evaluation sets (500 held-out test tickets) using a fixed random seed (`random_state=42`).
   - The classifier is trained **strictly on the training split**.
   - Held-out test tickets are evaluated **strictly on opening customer messages** (`customer_message`).
2. **Ground-Truth Labelling Procedure:**
   - Ground truth evaluation labels are generated using a rule-based labelling procedure based on human agent resolution actions (`agent_notes`).
   - *Transparency Note:* Evaluation labels are produced via an automated rule-based procedure, NOT a manual human audit.
3. **Held-Out Test Results (500 Tickets):**
   - **Overall Accuracy:** 80.20% (401 correct out of 500)
   - **Weighted Precision:** 83%
   - **Weighted F1-score:** 81%
   - **Delivery & Shipping:** 94% Precision, 80% Recall (0.86 F1)
   - **Account & Login:** 100% Precision, 100% Recall (1.00 F1)
   - **Product Enquiry:** 96% Precision, 93% Recall (0.95 F1)

---

## 💰 Business & Financial Framing

All numbers are derived directly from the source dataset and Support Operating Policy v3.2 (§3, §4):

1. **Planned Staffing Allocation:**
   - **₹9,00,000 / year headcount budget** (2 planned hires @ ₹4.5L/year each) is recommended to be reallocated from Billing to **Logistics** (where 28.4% of total demand exists).
2. **Current Operational Exposure:**
   - **SLA Breach Penalty Exposure:** **₹4,52,200** across 1,292 in-scope first-response breaches (@ ₹350 store credit per breach).
   - **Transfer Re-handling Cost Exposure:** **₹396,195** across 1,299 recorded helpdesk hand-offs (@ ₹305 per transfer).
3. **Avoidable Misrouting Cost Exposure:**
   - Chatbot intake errors routed 996 Logistics tickets into Billing.
   - These misrouted tickets generated 535 unnecessary transfers (**₹1,63,175**) and 347 SLA breaches (**₹121,450**), creating **₹2,86,635 in avoidable misrouting cost exposure**.

---

## ⚠️ Known Limitations & Assumptions

1. **Intake Text Ambiguity:** Extremely short 3-word customer messages (e.g. *"paid no update"*) lack order status context and cannot distinguish payment gateway failures from courier dispatch delays without live API integration into the warehouse order database.
2. **Rule-Based Evaluation Labels:** Validation ground truth was generated using an automated rule-based procedure on agent notes rather than a manual human audit, introducing minor label noise (~5-10%).
3. **Legacy Transfers Data:** Pre-Sept 2025 Freshdesk tickets (`legacy_fd`) lack transfer count history (`transfers` = null per §9), so transfer costs are evaluated on helpdesk records.

---

## 🤖 AI Models & API Execution Costs

- **Model Used:** Scikit-learn Logistic Regression + TF-IDF Vectorizer + Python Regex Domain Rules.
- **Execution Cost:** **₹0.00 per run** (runs locally in ~1.2s).
- **Monthly Cost at ~650 tickets/week (~2,800 tickets/month):** **₹0.00 / month** locally. (If run via cloud LLM API @ 300 tokens/ticket, monthly cost is ~$0.13 / month (~₹11 INR)).
