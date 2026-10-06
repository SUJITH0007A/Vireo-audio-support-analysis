"""
Validator module for Vireo Audio Support Ticket Classifier.
Performs train/test evaluation using a fixed random seed (random_state=42).
Uses a rule-based ground-truth labelling procedure on held-out evaluation tickets.
Evaluates predictions strictly on customer opening text (customer_message).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

def assign_rule_based_ground_truth(row):
    """
    Rule-based ground-truth labelling procedure for retrospective evaluation.
    Evaluates human agent resolution actions (agent_notes) and customer message.
    Note: This is an automated rule-based proxy label procedure, NOT a manual human audit.
    """
    notes = str(row.get('agent_notes', '')).lower()
    msg = str(row.get('customer_message', '')).lower()
    text = notes + " " + msg
    
    # 1. Warranty & RMA
    if any(k in notes for k in ['rma', 'service centre', 'service center', 'warranty claim', 'claim number']) or 'rma' in text:
        return 'Warranty & RMA'
        
    # 2. Returns & Refunds
    if any(k in notes for k in ['reverse pickup', 'pkp', 'return qc', 'return initiated', 'return requested', 'pickup missed', 'return label']) or ('return' in msg and ('pickup' in msg or 'refund' in msg or 'courier' in msg)):
        return 'Returns & Refunds'

    # 3. Delivery & Shipping
    if any(k in notes for k in ['courier', 'shipment', 'delivery', 'dlvry', 'tracking', 'awb', 'rto', 'shipped', 'transit', 'not delivered', 'dispatch', 'pincode', 'address update']):
        return 'Delivery & Shipping'
    if any(k in msg for k in ['not delivered', 'shipment', 'tracking', 'courier', 'where is my order', 'money gone', 'paid', 'deducted', 'deliver']) and not ('double' in msg or 'gateway' in msg or 'invoice' in msg):
        return 'Delivery & Shipping'

    # 4. Genuine Billing & Payments
    if any(k in notes for k in ['double payment', 'charged twice', 'failed payment', 'gateway error', 'invoice request', 'price adjustment', 'coupon code', 'payment failed']):
        return 'Billing & Payments'
    if any(k in msg for k in ['charged twice', 'failed payment', 'invoice', 'double payment', 'coupon', 'price adjustment']):
        return 'Billing & Payments'

    # 5. Account & Login
    if any(k in notes for k in ['otp', 'login', 'logged out', 'locked out', 'password reset', 'account access']) or any(k in msg for k in ['otp', 'login', 'locked out', 'password']):
        return 'Account & Login'

    # 6. Product Enquiry
    if any(k in notes for k in ['pre-sales', 'compatibility', 'waterproof', 'shower', 'specs', 'specification', 'query answered']) or any(k in msg for k in ['compatible', 'shower', 'waterproof', 'specs']):
        return 'Product Enquiry'

    # 7. Hardware & Technical
    if any(k in notes for k in ['battery', 'charge', 'case light', 'power on', 'static', 'noise', 'sound', 'mic', 'volume', 'crackling', 'pair', 'bluetooth', 'connect', 'app crash', 'firmware']) or any(k in msg for k in ['battery', 'charge', 'static', 'noise', 'mic', 'volume', 'pair', 'bluetooth', 'connect', 'app', 'update']):
        return 'Hardware & Technical'

    # Fallback mapping
    orig = str(row.get('category', ''))
    if orig in ['Charging & Battery', 'Audio Quality', 'Connectivity', 'App & Firmware']:
        return 'Hardware & Technical'
    elif orig == 'Warranty & Repair':
        return 'Warranty & RMA'
    elif orig in ['Delivery & Shipping', 'Returns & Refunds', 'Billing & Payments', 'Account & Login', 'Product Enquiry']:
        return orig
    else:
        return 'Other / Unclear'

def run_train_test_validation(tickets_df, test_size=500, random_state=42):
    """
    Executes a train/test evaluation split:
    - Trains the classifier ONLY on the training split.
    - Evaluates predictions on a held-out test dataset of 500 tickets.
    - Evaluates using ONLY customer_message available at intake time.
    """
    from classifier import TicketClassifier
    
    df = tickets_df.copy()
    
    # Generate rule-based ground-truth proxy labels
    df['rule_ground_truth'] = df.apply(assign_rule_based_ground_truth, axis=1)
    
    # Train / Test split with fixed random seed
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )
    
    # Train classifier strictly on training set
    clf = TicketClassifier()
    clf.train(train_df)
    
    # Predict on held-out test set using ONLY customer_message
    predictions = []
    reasonings = []
    
    for idx, row in test_df.iterrows():
        cat, team, conf, reason = clf.predict_intake(row['customer_message'], row['channel'])
        predictions.append(cat)
        reasonings.append(reason)
        
    eval_df = test_df.copy()
    eval_df['predicted_category'] = predictions
    eval_df['eval_reasoning'] = reasonings
    eval_df['is_correct'] = eval_df['predicted_category'] == eval_df['rule_ground_truth']
    
    correct_cnt = eval_df['is_correct'].sum()
    total_cnt = len(eval_df)
    accuracy = correct_cnt / total_cnt
    
    report_dict = classification_report(
        eval_df['rule_ground_truth'],
        eval_df['predicted_category'],
        output_dict=True,
        zero_division=0
    )
    
    report_str = classification_report(
        eval_df['rule_ground_truth'],
        eval_df['predicted_category'],
        zero_division=0
    )
    
    labels = sorted(list(set(eval_df['rule_ground_truth']).union(set(eval_df['predicted_category']))))
    conf_matrix = confusion_matrix(eval_df['rule_ground_truth'], eval_df['predicted_category'], labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    
    cols_to_show = [c for c in ['ticket_id', 'customer_message', 'category', 'rule_ground_truth', 'predicted_category', 'eval_reasoning'] if c in eval_df.columns]
    error_analysis = eval_df[~eval_df['is_correct']][cols_to_show]
    
    # Export ground truth evaluation sample
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, 'ground_truth_sample.csv'), index=False)
    
    results = {
        'sample_size': total_cnt,
        'correct_predictions': int(correct_cnt),
        'incorrect_predictions': int(total_cnt - correct_cnt),
        'accuracy': round(accuracy, 4),
        'classification_report_dict': report_dict,
        'classification_report_str': report_str,
        'confusion_matrix_df': conf_matrix_df,
        'eval_df': eval_df,
        'error_analysis': error_analysis,
        'methodology_note': "500-ticket held-out evaluation sample using a rule-based ground-truth labelling procedure."
    }
    
    return results

if __name__ == '__main__':
    from data_loader import load_raw_data, clean_tickets_data
    t, a, c, o, p = load_raw_data()
    t_clean = clean_tickets_data(t)
    results = run_train_test_validation(t_clean, test_size=500, random_state=42)
    
    print(f"=== HELD-OUT VALIDATION RESULTS (Sample Size: {results['sample_size']}) ===")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Incorrect Predictions: {results['incorrect_predictions']}")
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
    print("\n--- CLASSIFICATION REPORT ---")
    print(results['classification_report_str'])
