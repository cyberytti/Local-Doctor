import pandas as pd

df1 = pd.read_csv("disease_sympts_prec_full.csv")
df2 = pd.read_csv("Diseases_Symptoms.csv")

df2 = df2.drop(columns=["Disease_Code", "Contagious", "Chronic"])
df2 = df2.rename(columns={
    "Name": "disease",
    "Symptoms": "symptoms",
    "Treatments": "precautions"
})

combined_df = pd.concat([df1, df2], ignore_index=True)
combined_df.to_csv("combined_disease_dataset.csv", index=False)


print(combined_df.columns)
print(combined_df.head())
