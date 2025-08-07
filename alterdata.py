import pandas as pd
data=pd.read_csv("Nike_dataset.csv")
#print(data.head())
data["Reviews_posted"]=2*data["Reviews_posted"]
data.to_csv("Nike_dataset.csv",index=False)