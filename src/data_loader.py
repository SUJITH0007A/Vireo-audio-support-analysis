"""
Data loader module for Vireo Audio Support Analysis.
Handles ingestion, cleaning, and joining of tickets, agents, customers, orders, and products.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'given files')

def load_raw_data(data_dir=DATA_DIR):
    """Load all 5 raw CSV datasets from the given files directory."""
    tickets_path = os.path.join(data_dir, 'tickets.csv.csv')
    agents_path = os.path.join(data_dir, 'agents.csv.csv')
    customers_path = os.path.join(data_dir, 'customer.csv.csv')
    orders_path = os.path.join(data_dir, 'orders.csv.csv')
    products_path = os.path.join(data_dir, 'products.csv.csv')
    
    tickets = pd.read_csv(tickets_path)
    agents = pd.read_csv(agents_path)
    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    products = pd.read_csv(products_path)
    
    return tickets, agents, customers, orders, products

def clean_tickets_data(tickets_df):
    """Clean ticket timestamps, handle missing values, and calculate key operational fields."""
    df = tickets_df.copy()
    
    # Parse dates
    df['created_at_dt'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['first_response_at_dt'] = pd.to_datetime(df['first_response_at'], errors='coerce')
    df['resolved_at_dt'] = pd.to_datetime(df['resolved_at'], errors='coerce')
    
    # Year-Month period
    df['month_year'] = df['created_at_dt'].dt.to_period('M').astype(str)
    
    # Response time & Resolution time (in hours)
    df['response_time_hrs'] = (df['first_response_at_dt'] - df['created_at_dt']).dt.total_seconds() / 3600.0
    df['resolution_time_hrs'] = (df['resolved_at_dt'] - df['created_at_dt']).dt.total_seconds() / 3600.0
    
    # SLA target per channel (Section 3 of Support Policy v3.2)
    # Chat: 15m (0.25h), Voice callback: 2h, Social: 4h, Email: 8h
    sla_targets_hrs = {
        'chat': 0.25,
        'voice': 2.0,
        'social': 4.0,
        'email': 8.0
    }
    df['sla_target_hrs'] = df['channel'].map(sla_targets_hrs)
    df['is_sla_breach'] = df['response_time_hrs'] > df['sla_target_hrs']
    df['sla_breach_penalty_inr'] = np.where(df['is_sla_breach'], 350.0, 0.0)
    
    # Direct Channel Contact Cost (Section 4 of Support Policy v3.2)
    channel_costs_inr = {
        'chat': 210.0,
        'email': 260.0,
        'voice': 520.0,
        'social': 240.0
    }
    df['channel_contact_cost_inr'] = df['channel'].map(channel_costs_inr)
    
    # Transfer Cost (Section 4: Rs 305 per transfer)
    df['transfers_count'] = df['transfers'].fillna(0)
    df['transfer_cost_inr'] = df['transfers_count'] * 305.0
    
    # Filter indicator for in-scope period (Jan 2025 - Jun 2026)
    df['in_scope'] = (df['created_at_dt'] >= '2025-01-01') & (df['created_at_dt'] <= '2026-06-30 23:59:59')
    
    return df

if __name__ == '__main__':
    t, a, c, o, p = load_raw_data()
    t_clean = clean_tickets_data(t)
    print("Data loaded cleanly.")
    print("Total tickets:", len(t_clean))
    print("In-scope tickets (Jan 2025 - Jun 2026):", t_clean['in_scope'].sum())
