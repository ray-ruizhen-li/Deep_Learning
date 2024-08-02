import pandas as pd

# create dummy features
def create_dummy_vars(df):
    #Separating target variable and other variables
    y= df.Attrition
    X= df.drop(columns = ['Attrition'])

    return X, y