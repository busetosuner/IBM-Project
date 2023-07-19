import pandas as pd
import numpy as np

#BINNING

def is_numeric(col):
    try:
        pd.to_numeric(col)
        return True
    except:
        return False

def make_bins(df, header):
    if is_numeric(df[header]):
        
        min_val = df[header].min()
        max_val = df[header].max()
        
        #Binning using np.linspace
        bins = np.linspace(min_val, max_val, num=4)
        labels = ['Low', 'Medium', 'High']
        
        # Perform binning       
        binned_column = pd.cut(df[header], bins=bins, labels=labels, include_lowest=True)
        
        # Updating df
        df[header + '_binned'] = binned_column
        
    else:
        # Categorical column
        top_10_values = df[header].value_counts().head(10).index.tolist()
        binned_column = df[header].apply(lambda x: x if x in top_10_values else 'Other')
        
        # Updating df
        df[header + '_binned'] = binned_column
    
    # Drop the original column
    df.drop(header, axis=1, inplace=True)

    return df, header + "_binned"
