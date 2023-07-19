
#BINNING

def binning(column, header):
    if is_numeric(column):
        
        min_val = column.min()
        max_val = column.max()
        
        #Binning using np.linspace
        bins = np.linspace(min_val, max_val, num=4)
        labels = ['Low', 'Medium', 'High']
        
        #Perform binning
        
        binned_column = pd.cut(column, bins=bins, labels=labels, include_lowest=True)
        
        #Updating df
        df_new[header + '_binned'] = binned_column
        
        
    else:
        #Categorical column için
        top_10_values = column.value_counts().head(10).index.tolist()
        binned_column = column.apply(lambda x: x if x in top_10_values else 'Other')
        
        
        #Updating df
        df_new[header + '_binned'] = binned_column



# Iterating over columns

for header in headers:
    column = df[header]
    binning(column, header)

print("Binning:", df_new)
