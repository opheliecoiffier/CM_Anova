"""
@authors: Ophélie Coiffier, Tanguy Lefort and Ibrahim Gaizi

In short: Usage of ANOVA on two datasets and presentation of
          the non-parametric permutation test.
"""

#################################
# Setup
# -------------------------------
# Packages needded

from download import download
import patsy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy as sp
from statsmodels.formula.api import ols
import os

sns.set_palette("colorblind")


#################################
# Download datasets
#################################

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'data')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

bicycle = "http://josephsalmon.eu/enseignement/datasets/bicycle_db.csv"
path_bicycle = os.path.join(results_dir, "bicycle.txt")

download(bicycle, path_bicycle, replace=False)

url = "http://josephsalmon.eu/enseignement/datasets/Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv"
path_target = os.path.join(results_dir, "Mesure_journaliere_Region_Occitanie_Polluants_Principaux.csv")
download(url, path_target, replace=False)

#################################
# Bicycle dataset
# -------------------------------
#
# Datasets recording informations about accidents involving bikes
# in France from 2005 to 2018.
#
# Clean-up the data and work selection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_bicycle = pd.read_csv(path_bicycle, sep=",", skiprows=2, converters={"heure": str})

###################################################
# Missing values, mistakes and variable selection
###################################################

df_bicycle = df_bicycle[["date", "mois", "jour", "heure", "departement",
                         "conditions atmosperiques", "gravite accident",
                         "sexe", "age", "existence securite"]]
df_bicycle["heure"].replace("", np.nan, inplace=True)
df_bicycle["age"].replace(["2004-2005", "2016-2017", "2006-2007", "2012-2013", "2013-2014",
                           '2005-2006', "2006-2007"], np.nan, inplace=True)
df_bicycle["existence securite"].replace("Inconnu", np.nan, inplace=True)
df_bicycle.dropna(inplace=True)
df_bicycle.rename(columns={"conditions atmosperiques": "conditions atmospheriques",
                           "gravite accident": "gravite_accident"}, inplace=True)

##############################
# Handling dates and time
##############################

df_bicycle.set_index(pd.to_datetime(df_bicycle["heure"] + "/00 "+
                            df_bicycle["date"],
                            format="%H/%M %Y-%m-%d"), inplace=True)
df_bicycle.drop(columns=["date"], inplace=True)
df_bicycle = df_bicycle[df_bicycle.index.year!=2018]

###############################
# Let's see what are the unique values inside our dataset now.

for i, name in enumerate(df_bicycle.columns):
    print("\n############################")
    print("Column", name)
    print(df_bicycle[df_bicycle.columns[i]].unique(), "\n")

print(df_bicycle.info)

##################################
# Data visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Let's take a look at some of our variables and
# their connections.
#
##################################
# Sexe repartition
##################################

plt.figure()
df_bicycle.sexe.value_counts().plot(kind='pie', labels=["Male", "Female"])
plt.title("Sexe repartition amongst the dataset.")
plt.show()

##################################
# Is it representative of the actual sexe repartition
# on bike-usages ? We don't have more data to confirm it
# or not, so we decided it was best to disregard sexe-related
# investigations.

#################################
# Accident gravity and security
#################################

print(pd.crosstab(df_bicycle['existence securite'], df_bicycle['gravite_accident'],
                  normalize='index', margins=True)*100)

#################################
# We can see that amongst our accidents with people wearing a helmet
# 57% of them have a small wound and 30% have been hospitalized.
# Whereas amongst the one with a belt as security 15% have a small wound
# and 70% have been hospitalized.
#
# Note that the securities and mutually excluding each other...

###################################################
# When are happening the most dangerous accidents?
###################################################

df_bicycle["heure"] = pd.to_numeric(df_bicycle["heure"])

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.violinplot(x="gravite_accident", y="heure", data=df_bicycle)
ax.set_xlabel('Accident gravity')
ax.set_ylabel('Hour')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize='small')
plt.ylim([0,24])
plt.title("When are happening the most dangerous accidents?")
plt.tight_layout()
plt.show()

######################################################
# Without any surprise, rush hours seem to be the most
# dangerous ones.


######################################################
# First ANOVA model: number of accidents by day
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_bicycle.groupby([df_bicycle.index.weekday, df_bicycle.index.hour])[
    'sexe'].count().unstack(level=0).plot()
plt.legend(labels=['Monday', "Tuesday", "Wednesday",
                   "Thursday", "Friday", "Saturday", "Sunday"])
plt.ylabel("Number of accidents"); plt.xlabel("Hours")
plt.savefig(os.path.join(script_dir, "images", "number_accidents_day.pdf"))
plt.show()

######################################################
# We can clearly see two trends: one for the work days
# and another for the weekend. As for the hours, we retrieve
# our previous conclusion about the rush hours.
#
# Let's test :math:`H_0:\mu_{monday}=\mu_{tuesday}=\dots=\mu_{sunday}`
# against :math:`H_1:` one distribution of the accidents for a day have a different
# mean than another one. 
# For that we first take a look at the distribution of each number of accidents by day
# using a violin plot.

data_days = df_bicycle.to_period("M")
data_days = data_days.pivot_table(index=data_days.index, columns='jour', aggfunc='size')

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
data_days = pd.DataFrame({"days": days*int(data_days.shape[0]),
                         "number_accident": data_days.stack().values})

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.violinplot(x="days", y="number_accident", data=data_days)
ax.set_ylabel("Number of accidents")
ax.set_xlabel('Days')
plt.title("Distribution of the number of accidents by week days.")

########################################################
# Let's make the ANOVA with :math:`\alpha=0.05` and
# check the residuals normality assumption.

lm_days = ols('number_accident ~ C(days)', data=data_days).fit()
anova_days = sm.stats.anova_lm(lm_days, typ=2)
print(anova_days)

resid_days = lm_days.resid
fig, ax = plt.subplots()
_, (__, ___, r) = sp.stats.probplot(resid_days, plot=ax, fit=True)

########################################################
# The p⁻value is way below :math:`\alpha`. We can reject :math:`H_0`.
# In conclusion we can say that there is an individual effect amongst the days.
# We already saw that there is at least the weekday/weekend effect.
# 
# The normality of the errros can also be assumed (for the most part,
# the observed values fit to the theoretical quantiles).
#
######################################################################
# Second ANOVA: Morning or afternoon, when should we be more careful?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let's only consider the accidents happening during the morning (from 1am to 1pm)
# and the ones happpening during the afternoon (from 2pm to 8pm).
# We want to test :math:`H_0: \mu_{morning}=\mu_{afternoon}` ie there is no
# individual effect from each time-duration against the hypothesis that there is one.

heures = df_bicycle.heure.isin(["6", "7", '8', '9', '10', "11", '12', '13', "14", "15", "16", "17", "18", "19", "20"])
df_all = df_bicycle[heures]
df_all = df_all.copy()
df_all["daytime"] = df_all.heure.isin(["6", "7", '8', '9', '10', "11", '12', "13"])
df_all['daytime'].replace(True, "matin", inplace=True)
df_all['daytime'].replace(False, "apres_midi", inplace=True)

data = df_all.to_period("M")
data = data.pivot_table(index=data.index, columns='daytime', aggfunc='size')
data.plot()
plt.ylabel("Number of accidents")
plt.xlabel("Months")
plt.legend(labels=['afternoon', "morning"])
plt.tight_layout()
plt.show()

##############################
# The data is very intricated but there seems to be more
# accidents during the afternoon, let's take a look at the distributions.

data = pd.DataFrame({"accident_number": data.stack().values,
                     "daytime": ['6-13h', '14-20h']*int(data.shape[0])})

fig, ax = plt.subplots(figsize=(8,5))
ax = sns.violinplot(x="daytime", y="accident_number", data=data)
ax.set_ylabel("Number of accidents")
ax.set_xlabel('Daytime')
plt.title("Number of accidents amongst the morning of afternoon by month")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize='small')

plt.tight_layout()
plt.show()

##################################
# Let's perform the actual test and check the residuals assumption
# with :math:`\alpha=0.05`.

lm_acc_day = ols('accident_number ~ C(daytime)', data=data).fit()
anova_acc_day = sm.stats.anova_lm(lm_acc_day, typ=2)
print(anova_acc_day)

resid_acc = lm_acc_day.resid
fig, ax = plt.subplots()
_, (__, ___, r) = sp.stats.probplot(resid_acc, plot=ax, fit=True)

#####################################
# The normality of the residuals can be assumed and the 
# ANOVA p-value is under our threshold so there is an indivual
# effect from our groups. Considering our visualizations, we can
# conclude that the number of accidents during the afternoon is,
# in average, significativvely greater than during the morning.

######################################
# Permutation test
# ------------------------------------
# 
# Monte-Carlo based non-parametric method.
# Simulation of the over-simplified medical protocol described in
# the beamer presentatioon. We test the effect of the treatment (B) against
# a control group (A).

class Treatment_test_simulation():

    def __init__(self, nb_patients, mean_A=None, mean_B=None, nb_permut=None, setseed=True):
        if (nb_patients % 2) != 0:
            nb_patients += 1  # only even number is easier
        self.n = nb_patients
        if mean_A is None:  # A=control ie gaussian mean effect is 3
            mean_A = 3 
            # A : miracle drug against COVID without memory in the system
        if mean_B is None:
            mean_B = 7  # B = test ie exponential mean effect is 7
        self.mean_A = mean_A
        self.mean_B = mean_B

        if nb_permut is None:  # alright for large numbers
            nb_permut = 100 * nb_patients
        self.nb_permut = nb_permut
        self.setseed = setseed
        if setseed:
            np.random.seed(11235813)

        self.group_A = np.random.normal(size=int(self.n/2),
                                        loc=self.mean_A)
        self.group_B = np.random.exponential(size=int(self.n/2),
                                             scale=self.mean_B)  # 1/lambda in python

    @staticmethod
    def compute_stat(rowtab):
        res = np.mean(rowtab[1, :]) - np.mean(rowtab[0, :])
        return res

    def permutation_test(self):
        J = self.nb_permut
        if self.setseed:
            rng = np.random.default_rng(seed=11235813)
        storage = np.zeros(J)
        row_tab = np.vstack((self.group_A, self.group_B))
        for j in range(J):
            data = row_tab.copy()
            rng.shuffle(data.reshape(-1), axis=0)
            storage[j] = self.compute_stat(data)
        return storage

    def plot_hist(self, storage, show=False):
        plt.figure()
        sns.distplot(storage)
        if show:
            plt.show()

    def answer_test(self, storage):
        row_tab = np.vstack((self.group_A, self.group_B))
        ref_ = self.compute_stat(row_tab)
        nb_over_ref = np.sum(storage >= ref_)
        return nb_over_ref / self.nb_permut, ref_

    def perform_test(self, plot=True):
        stor = self.permutation_test()
        pval, ref_ = self.answer_test(stor)
        if plot:
            plt.close("all")
            self.plot_hist(stor)
            plt.gca()
            plt.axvline(x=ref_, linewidth=2, color='r',
                        label=r'$reference\ value: \hat\mu_B - \hat\mu_A$')
            plt.title(r"Histogram of the density for the statistic $\hat\mu_B - \hat\mu_A$")
            plt.ylabel("density")
            plt.xlabel("Test values for each permutation")
            plt.legend()
            name = "rejet.pdf" if pval < .05 else "not_rejet.pdf"
            plt.savefig(os.path.join(script_dir, 'images', name))
            plt.show()
        return pval, ref_


##################################
# Let's perform it on two cases, the first where we reject :math:`H_0: \mu_A=\mu_B`
#  against :math:`H_1:\mu_A\geq \mu_B`

test_reject = Treatment_test_simulation(50, setseed=True, nb_permut=300)
pval, _ =test_reject.perform_test()
print("The p-value is {:.4f} < 5%".format(pval))

###################################
# We conclude that the treatment has a positive effect, there is a significant difference
# between the averages oof the distribution.

test_not_reject = Treatment_test_simulation(50, mean_A = 2, mean_B = 2.5,
                                             setseed=True, nb_permut=300)
pval, _ = test_not_reject.perform_test()
print("The p-value is {:.4f} > 5%".format(pval))

##################################
# We can"t say that there is a significative difference between the tested group and
# the control one.



####################################
# Pollution dataset
#----------------------------------
#
# Datasets recording informations about the concentration of pollution in few cities from 2017 to 2018.
#
# Work selection on the variable
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df = pd.read_csv(path_target)
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
plt.savefig(os.path.join(script_dir, "images", "O3_by_city.pdf"))
plt.show()

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

df_mois = pd.read_csv(path_target, index_col="date_debut")
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
plt.plot(x, y2, label="Nimes")
plt.plot(x, y3, label="Montpellier")
plt.plot(x, y4, label="Castres")
plt.title("Mean of O3 concentration by month")
plt.xlabel("Month")
plt.ylabel("Concentration of O3")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "images", "Mean_of_O3.pdf"))
plt.show()

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
plt.savefig(os.path.join(script_dir, "images", "Verification_of_residues.pdf"))
plt.show()


###############################################################################################
# We can see that the residuals follow the normal distribution thanks to the Probability Plot.
# We have the hypothesis H_0 :math:'\mu_{Montpellier}=\mu_{Castres}=\mu_{Tarbes}=\mu_{Nimes}' and :math:'\alpha=0.05'
# The p-value is lower than :math:'\alpha so we reject H_0'. 
# These cities haven't the same mean of O3 pollution.
################################################################################################
