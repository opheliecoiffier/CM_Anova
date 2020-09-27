# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from download import download
import statsmodels.api as sm
import scipy as sp
from statsmodels.formula.api import ols
import numpy as np
sns.set_palette("colorblind")
sns.set()

####################################
# Download datasets
####################################

url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = "./Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
download(url, path_target, replace=False)

####################################
# Pollution dataset
#----------------------------------
#
# Datasets recording informations about the concentration of pollution in few cities from 2017 to 2018.
#
# Work selection on the variable
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df = pd.read_csv('Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv')
df= df.loc[:, ["polluant", 'nom_com', "valeur_originale"]]
#creation du tableau ne contenant que ces villes et O3 comme polluant
df = df.loc[df["nom_com"].isin(["MONTPELLIER","NIMES", "TARBES", "CASTRES"])]
df = df.loc[df["polluant"].isin(["O3"])]

#######################################################################################
# Descriptive analysis of the pollution by O3 in Nimes, Tares, Montpellier and Castres.
########################################################################################

fig = sns.catplot(x=df.columns[1], y="valeur_originale",
         data=df, kind="violin", legend=False)
plt.title("O3 by city")
plt.xlabel("cities")
plt.ylabel("Concentration of O3")
plt.tight_layout()
fig.savefig('O3_by_city.pdf')

##################################################################################
# We can see that the concentration of O3 is higher in Nimes than the others.
# The densiest concentration is located between 40 and 80 ug.m^3 for all cities.
#####################################################################################

###################################################################
# Descriptive analysis of the pollution concentration
# We can see the mean, the standard error, the min and max, the 1st and 3rd quantile.
#################################################################

print(df.describe())


########################################################
# Look at the mean of O3 pollution by month and by city
########################################################

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
   
fig = plt.figure()
plt.plot(x, y1, label="Tarbes")
plt.plot(x, y2, label="Nîmes")
plt.plot(x, y3, label="Montpellier")
plt.plot(x, y4, label="Castres")
plt.title("Mean of O3 concentration by month")
plt.xlabel("Month")
plt.ylabel("Concentration of O3")
plt.legend()
plt.tight_layout()
fig.savefig("Mean_of_O3.pdf")

##############################################################
# We can see that these curves are roughly the same profil : 
# there is an augmentation between January and July, then the pollution declines.
# Nimes is the city who has the more important mean of concentration. It's located in July.
# Tarbes is the city who has the smallest mean of concentration. It's located in November.
##################################################################

##############################################
# ANOVA model : O3 concentration by cities
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

poll = ols('valeur_originale ~ C(nom_com)',data=df).fit()
print(poll.summary())
pollution = sm.stats.anova_lm(poll, typ=2) 
print(pollution)
fig, ax = plt.subplots()
_, (__, ___, r) = sp.stats.probplot(poll.resid, plot=ax, fit=True)
fig.savefig('Verification_of_residues.pdf')

###############################################################################################
# We can see that the residuals follow the normal distribution thanks to the Probability Plot.
# We have the hypothesis H_0 : math:'\mu_{Montpellier}=\mu_{Castres}=\mu_{Tarbes}=\mu_{Nimes}' and math:'\alpha=0.05'
# The p-value is lower than math:'\alpha so we reject H_0'. 
# These cities haven't the same mean of O3 pollution.
################################################################################################
