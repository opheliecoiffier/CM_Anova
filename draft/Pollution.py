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
#%%
import numpy as np
df_mois = pd.read_csv('Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv', index_col="date_debut")
df_mois = df_mois.loc[:, ["polluant", 'nom_com', "valeur_originale"]]
df_mois = df_mois.loc[df_mois["nom_com"].isin(["MONTPELLIER","NIMES", "TARBES", "CASTRES"])]
df_mois = df_mois.loc[df_mois["polluant"].isin(["O3"])]
df_mois.set_index(pd.to_datetime(df_mois.index, format = "%Y-%m-%d"), inplace=True)

tab = pd.DataFrame(columns=["mois","O3"])
tab = df_mois.loc[df_mois["nom_com"].isin(["TARBES"])]
y1 = np.ones(12)
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for i in x :
    tab1 = tab.loc[tab.index.month==i]
    y1[i-1] = np.mean(tab1['valeur_originale'])

tab2 = tab = df_mois.loc[df_mois["nom_com"].isin(["NIMES"])]
y2 = np.ones(12)
for i in x :
    tab2 = tab.loc[tab.index.month==i]
    y2[i-1] = np.mean(tab2['valeur_originale']) 

tab3 = tab = df_mois.loc[df_mois["nom_com"].isin(["MONTPELLIER"])]
y3 = np.ones(12)
for i in x :
    tab3 = tab.loc[tab.index.month==i]
    y3[i-1] = np.mean(tab3['valeur_originale']) 

tab4 = tab = df_mois.loc[df_mois["nom_com"].isin(["CASTRES"])]
y4 = np.ones(12)
for i in x :
    tab4 = tab.loc[tab.index.month==i]
    y4[i-1] = np.mean(tab4['valeur_originale']) 
    
plt.plot(x, y1, label="Tarbes")
plt.plot(x, y2, label="Nîmes")
plt.plot(x, y3, label="Montpellier")
plt.plot(x, y4, label="Castres")
plt.title("mean of O3 concentration by month")
plt.xlabel("month")
plt.ylabel("Concentration of O3")
plt.legend()
plt.tight_layout()