import pandas as pd 
from process_data import PreProcessData, adjust_prediction
from train_test_split import (get_claim_train_test, get_endorsement_train_test)
from train_model import get_claims_xgb_model, get_endorsement_xgb_model
from predict_data import PredictProbability

class GetClaims:
    def __init__(self, filepath):
        self.training_file_path = filepath

    def process_and_predict(self):
        #generate feature rich model
        processData = PreProcessData(self.training_file_path)
        pre_processed_df = processData.general_data_cleaning()

        #generate dataframe for early claims
        claims_df = processData.generate_claims_dataset(pre_processed_df)

        #generate dataframe for early endorsements
        endorsement_df = processData.generate_endorsement_dataset(pre_processed_df)

        #get train/test datasets
        claims_X_train, claims_X_test, claims_y_train, claims_y_test = get_claim_train_test(claims_df)

        endorse_X_train, endorse_X_test, endorse_y_train, endorse_y_test = get_endorsement_train_test(endorsement_df)

        #train early claims model
        xgb_claims_model = get_claims_xgb_model(claims_X_train, claims_y_train)

        #train endorsement model
        xgb_endorse_model = get_endorsement_xgb_model(endorse_X_train, endorse_y_train)

        #predict claims probability on entire dataset
        remove_vals = ['policy_id','has_claim_first_14','has_claim_last_14','max_endorse_count']
        predict_outcome = PredictProbability(xgb_claims_model, claims_df, remove_vals)

        final_claims_df = predict_outcome.run_model()
        final_claims_df = final_claims_df.rename(columns={'prediction':'claim_prediction','probability':'claim_probability'})

        #predict endorsement probability on entire dataset
        remove_endorse_vals = ['policy_id','has_endorsement_first_7','endorsement_date_diff','endorse_bucket','written_premium_total','written_premium_incr',
                        'incurred_loss_+_dcc_total','claim_coverage','claim_status','claim_with_payment','feature_status','feature_with_payment']
        predict_outcome = PredictProbability(xgb_endorse_model, endorsement_df, remove_endorse_vals)

        final_endorse_df = predict_outcome.run_model()
        final_endorse_df = final_endorse_df.rename(columns={'prediction':'endorsement_prediction','probability':'endorsement_probability'})

        #reduce final model dfs to just the prediction and probabilities
        final_claims_df = final_claims_df[['policy_id','claim_probability','claim_prediction']]

        final_endorse_df = final_endorse_df[['policy_id','endorsement_probability','endorsement_prediction']]

        final_df = pd.merge(pre_processed_df, final_claims_df, on='policy_id', how='inner')

        final_df = pd.merge(final_df, final_endorse_df, on='policy_id', how='inner')

        return final_df


if __name__ == '__main__':
    d = GetClaims('src/data/responsive_auto_raw_csv.csv')

    df = d.process_and_predict()

    #write final dataframe to file
    file_name = 'src/output/responsive_auto_predictions_v4.csv'
    df.to_csv(file_name, encoding='utf-8', index=False)
    