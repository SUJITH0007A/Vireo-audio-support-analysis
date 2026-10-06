# SCREEN RECORDING SCRIPT — VIREO AUDIO SUPPORT ANALYSIS
**Target Duration:** Maximum 3 Minutes (180 Seconds)  
**Format:** Live Screen & Code Walkthrough (No Slides)

---

### Timed Script Breakdown

#### 0:00 – 0:25 | What the Tool Does (25s)
> *"Hello! Today I'm presenting the Vireo Audio Support Ticket Analysis Tool. Vireo Audio was considering allocating two new support hires—a 9 Lakh rupees annual budget—to the Billing team because their intake bot reported Billing as a top queue.*  
> *We built a practical AI-assisted classification tool that ingests 11,780 customer tickets, re-categorizes them based on customer intent at intake, quantifies team workload, and provides defensible staffing recommendations."*

#### 0:25 – 0:55 | Initial Approach & Why It Was Improved (30s)
> *"Initially, evaluating raw intake category tags suggested Billing was drowning. However, inspecting agent notes revealed that over 40% of tickets assigned to Billing were actually delivery delays where customers mentioned payment words like 'paid' or 'money deducted'.*  
> *We improved our approach by creating a hybrid classifier combining TF-IDF vectorization, Logistic Regression, and domain rules. Crucially, our live intake model uses ONLY customer opening messages without leaking agent notes, while historical audits analyze full resolution context."*

#### 0:55 – 1:45 | Live Demo: Web Dashboard & CLI (50s)
> *(Screen: Show Streamlit Web Application `app.py`)*  
> *"Let's run the tool live. Here in the Streamlit web dashboard, we see the executive overview across 11,641 in-scope tickets. Looking at the team workload chart, under old intake routing, Billing appeared to have 20.8% of volume.*  
> *Under AI re-classification, Logistics is actually the largest queue by far at 28.4% of total demand (3,309 tickets), with a median resolution time of 19.1 hours and an SLA breach rate of 17.1%. In the interactive sandbox, entering 'paid on 19 jun, still waiting for package' correctly predicts 'Delivery & Shipping' routed to Logistics."*

#### 1:45 – 2:20 | Validation Methodology & Results (35s)
> *"We evaluated our classifier using an 80/20 train/test split with a fixed random seed (random_state=42). Evaluated on 500 held-out test tickets using ONLY customer opening text against a rule-based ground-truth labelling procedure, the model achieved 80.2% overall accuracy, 83% weighted precision, and 94% precision on Delivery & Shipping.*  
> *We explicitly document that evaluation labels were generated via automated rule-based procedures rather than manual human audits."*

#### 2:20 – 2:50 | Staffing Recommendation & Financial Exposure (30s)
> *"Based on actual workload data, we push back on giving 2 hires to Billing. We recommend re-allocating both planned hires (9 Lakh budget) to Logistics.*  
> *Rather than claiming guaranteed savings, we show that this reallocation addresses the 3,300-ticket bottleneck in Logistics while eliminating up to 2.86 Lakhs in avoidable misrouting re-handling costs and SLA breach credit exposure."*

#### 2:50 – 3:00 | Limitations & Exclusions (10s)
> *"As a limitation, extremely short 3-word messages like 'paid no update' remain ambiguous at intake without querying live order tracking APIs. We deliberately left out complex cloud microservices to keep the tool simple, zero-cost, and instantly runnable on any machine. Thank you!"*
