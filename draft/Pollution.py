# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from download import download

url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = "./Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
download(url, path_target, replace=False)

#%%
df = pd.read_csv('Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv')
df= df.loc[:,["polluant", 'nom_com', "valeur_originale"]]
#creation du tableau ne contenant que ces villes et O3 comme polluant
df = df.loc[df["nom_com"].isin(["MONTPELLIER","NIMES", "TARBES", "CASTRES"])]
df = df.loc[df["polluant"].isin(["O3"])]
print(df["polluant"])

#violins : descriptive analysis
sns.catplot(x=df.columns[1], y="valeur_originale",
         data=df, kind="violin", legend=False)
plt.title("O3 par ville")
plt.legend(loc=1)
plt.tight_layout()

print(df.head())

#%%
#descriptive analysis
print(df.describe())

#%%
#Anova