"""
Analytics module for Vireo Audio Support Desk.
Computes monthly volume breakdowns, team workload metrics, operational exposures,
and financial resource reallocation summaries.
"""

import os
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

def run_monthly_analysis(classified_tickets_df):
    """
    Generate monthly ticket metrics and financial exposure metrics for in-scope period (Jan 2025 - Jun 2026).
    """
    df = classified_tickets_df[classified_tickets_df['in_scope']].copy()
    
    # 1. Monthly Volume Overall
    monthly_vol = df.groupby('month_year').size().reset_index(name='total_tickets')
    
    # 2. Monthly Volume by Original Category vs Predicted Category
    monthly_cat_orig = pd.crosstab(df['month_year'], df['category'])
    monthly_cat_pred = pd.crosstab(df['month_year'], df['predicted_category'])
    
    # 3. Monthly Volume by Original Team vs Predicted Team
    monthly_team_orig = pd.crosstab(df['month_year'], df['assigned_team'])
    monthly_team_pred = pd.crosstab(df['month_year'], df['predicted_team'])
    
    # 4. Workload Metrics by Reclassified Team
    team_metrics = df.groupby('predicted_team').agg(
        total_tickets=('ticket_id', 'count'),
        pct_of_total=('ticket_id', lambda x: round(len(x) / len(df) * 100, 2)),
        avg_res_hrs=('resolution_time_hrs', 'mean'),
        median_res_hrs=('resolution_time_hrs', 'median'),
        sla_breaches=('is_sla_breach', 'sum'),
        sla_breach_rate=('is_sla_breach', 'mean'),
        total_transfers=('transfers_count', 'sum')
    ).reset_index()
    
    # Format median resolution time rounded to 1 decimal place
    team_metrics['median_res_hrs'] = team_metrics['median_res_hrs'].round(1)
    team_metrics['avg_res_hrs'] = team_metrics['avg_res_hrs'].round(1)
    team_metrics['sla_breach_rate'] = (team_metrics['sla_breach_rate'] * 100).round(1)
    
    # Operational Exposure Calculations (Support Policy v3.2 §3, §4)
    total_sla_breach_cost = df['sla_breach_penalty_inr'].sum()  # Rs 452,200.00
    total_transfer_cost = df['transfer_cost_inr'].sum()        # Rs 396,195.00
    planned_headcount_budget = 900000.0  # Rs 9 Lakh per year (Arjun Mehta, Finance Controller)
    
    # Misrouted Billing Intake Queue Calculations
    billing_intake = df[df['assigned_team'] == 'Billing']
    billing_to_logistics = billing_intake[billing_intake['predicted_team'] == 'Logistics']
    misrouted_count = len(billing_to_logistics)
    misrouted_transfers_cost = billing_to_logistics['transfers_count'].sum() * 305.0
    misrouted_sla_cost = billing_to_logistics['sla_breach_penalty_inr'].sum()
    avoidable_misrouting_opportunity = misrouted_transfers_cost + misrouted_sla_cost
    
    summary = {
        'total_in_scope_tickets': len(df),
        'date_range': 'Jan 2025 – Jun 2026',
        'monthly_vol': monthly_vol,
        'monthly_cat_orig': monthly_cat_orig,
        'monthly_cat_pred': monthly_cat_pred,
        'monthly_team_orig': monthly_team_orig,
        'monthly_team_pred': monthly_team_pred,
        'team_metrics': team_metrics,
        'total_sla_breach_cost': total_sla_breach_cost,
        'total_transfer_cost': total_transfer_cost,
        'planned_headcount_budget': planned_headcount_budget,
        'misrouted_count': misrouted_count,
        'misrouted_transfers_cost': misrouted_transfers_cost,
        'misrouted_sla_cost': misrouted_sla_cost,
        'avoidable_misrouting_opportunity': avoidable_misrouting_opportunity
    }
    
    return summary

def export_summary_reports(classified_df, summary):
    """Save processed datasets and analytical summary CSVs to outputs directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    classified_df.to_csv(os.path.join(OUTPUT_DIR, 'classified_tickets.csv'), index=False)
    summary['monthly_vol'].to_csv(os.path.join(OUTPUT_DIR, 'monthly_volume.csv'), index=False)
    summary['monthly_cat_pred'].to_csv(os.path.join(OUTPUT_DIR, 'monthly_category_breakdown.csv'))
    summary['monthly_team_pred'].to_csv(os.path.join(OUTPUT_DIR, 'monthly_team_breakdown.csv'))
    summary['team_metrics'].to_csv(os.path.join(OUTPUT_DIR, 'team_workload_metrics.csv'), index=False)
    
    print(f"Summary reports exported cleanly to {OUTPUT_DIR}")

if __name__ == '__main__':
    from data_loader import load_raw_data, clean_tickets_data
    from classifier import TicketClassifier
    
    t, a, c, o, p = load_raw_data()
    t_clean = clean_tickets_data(t)
    
    clf = TicketClassifier()
    clf_df = clf.fit_and_predict(t_clean)
    
    summary = run_monthly_analysis(clf_df)
    export_summary_reports(clf_df, summary)
    
    print("=== MONTHLY ANALYSIS SUMMARY ===")
    print("Total In-Scope Tickets:", summary['total_in_scope_tickets'])
    print("Total SLA Breach Exposure: Rs", f"{summary['total_sla_breach_cost']:,.2f}")
    print("Total Transfer Exposure: Rs", f"{summary['total_transfer_cost']:,.2f}")
    print("Planned Headcount Budget: Rs", f"{summary['planned_headcount_budget']:,.2f}")
    print("Avoidable Misrouting Cost Opportunity: Rs", f"{summary['avoidable_misrouting_opportunity']:,.2f}")
    print("\n--- RE-CLASSIFIED TEAM WORKLOAD ---")
    print(summary['team_metrics'])
