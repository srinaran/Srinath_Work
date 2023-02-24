from sklearn.model_selection import train_test_split # Import train_test_split function


remove_claims_cols = ['policy_id','has_claim_first_14','has_claim_last_14','max_endorse_count']
target_claims_col = 'has_claim_first_14'

remove_endorse_cols = ['policy_id','has_endorsement_first_7','endorsement_date_diff',
                        'endorse_bucket','written_premium_total','written_premium_incr',
                        'incurred_loss_+_dcc_total','claim_coverage','claim_status',
                        'claim_with_payment','feature_status','feature_with_payment']
target_endorse_col = 'has_endorsement_first_7'                    

def get_claim_train_test(df):
    x_vals = [col for col in df.columns if col not in remove_claims_cols]
    
    X = df[x_vals]
    y = df[target_claims_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
    
    return X_train, X_test, y_train, y_test


def get_endorsement_train_test(df):
    x_vals = [col for col in df.columns if col not in remove_endorse_cols]
    
    X = df[x_vals]
    y = df[target_endorse_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
    
    return X_train, X_test, y_train, y_test
