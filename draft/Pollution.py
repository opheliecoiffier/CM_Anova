# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from download import download
import statsmodels.api as sm
import scipy as sp
from statsmodels.formula.api import ols
sns.set_palette("colorblind")
sns.set()

url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = "./Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
download(url, path_target, replace=False)

#%%
#new data
df = pd.read_csv('Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv')
df= df.loc[:, ["polluant", 'nom_com', "valeur_originale"]]
#creation du tableau ne contenant que ces villes et O3 comme polluant
df = df.loc[df["nom_com"].isin(["MONTPELLIER","NIMES", "TARBES", "CASTRES"])]
df = df.loc[df["polluant"].isin(["O3"])]

#violins : descriptive analysis
sns.catplot(x=df.columns[1], y="valeur_originale",
         data=df, kind="violin", legend=False)
plt.title("O3 by city")
plt.xlabel("cities")
plt.ylabel("Concentration of O3")
plt.legend(loc=1)
plt.tight_layout()

#%%
#descriptive analysis
print(df.describe())

#%%
#Anova
poll = ols('valeur_originale ~ C(nom_com)',data=df).fit()
print(poll.summary())
pollution = sm.stats.anova_lm(poll, typ=2) 
print(pollution)
fig, ax = plt.subplots()
_, (__, ___, r) = sp.stats.probplot(poll.resid, plot=ax, fit=True)
