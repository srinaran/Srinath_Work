import pandas as pd 

class PredictProbability:
    def __init__(self, model, df, remove_x_vals):
        self.model = model
        self.df = df
        self.remove_x_vals = remove_x_vals
        self.run_model

    def run_model(self):
        x_vals = [col for col in self.df.columns if col not in self.remove_x_vals]

        X = self.df[x_vals]
        #y = self.df[self.target_val]
        
        predictions = self.model.predict(X)

        probabilities = self.model.predict_proba(X)[:, 1]

        # append predictions column and probability column to end of test df
        self.df['prediction'] = predictions

        self.df['probability'] = probabilities

        return self.df