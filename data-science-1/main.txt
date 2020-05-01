import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

from IPython.core.pylabtools import figsize
figsize(12, 8)
sns.set()

np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})
df = dataframe.copy()

df.head()

df_info = pd.DataFrame({'mean': [df.normal.mean(),df.binomial.mean()], 
              'std': [df.normal.std(),df.binomial.std()],
              'var': [df.normal.var(),df.binomial.var()],
              'var': [round(df.normal.var(),2), round(df.binomial.var(),2)], 
              'max': [max(df.normal), max(df.binomial)], 
              'min': [min(df.normal),min(df.binomial)]},
                index = ['normal','binominal' ])
                
df_info

# Sua análise da parte 1 começa aqui.abs
sns.distplot(df['normal'], label="Normal",  hist_kws={"alpha": 0.2})
plt.show()

sns.distplot(df['binomial'], bins=range(6,36), kde=True, label = "Binominal",  hist_kws={"alpha": 0.9})
plt.show()

#Using the theorical distribution
q1_norm, q2_norm, q3_norm = [sct.norm.ppf(x, loc = 20, scale = 4) for x in [0.25, 0.5, 0.75]]
q1_binom, q2_binom, q3_binom = [sct.binom.ppf(x, 100, 0.2) for x in [0.25, 0.5, 0.75]]
diff =(q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom) 

print("Q% normal", q1_norm, q2_norm, q3_norm)
print("Q% binomial", q1_binom, q2_binom, q3_binom)
print('Difference', diff)

#Using the dataframe
quantiles = [df.quantile(q = x) for x in [0.25, 0.5, 0.75]] #get quantiles of the dataframe
diff= tuple(([round(quantiles[i][0]- quantiles[i][1],3) for i in range(3)])) #subtract to get the difference

print("Q% normal", [quantiles[i][0] for i in range(3)])
print("Q% binomial", [quantiles[i][1] for i in range(3)])
print('Difference', diff)

def q1():
    quantiles = [df.quantile(q = x) for x in [0.25, 0.5, 0.75]] #get quantiles of the dataframe
    aux = ([round(quantiles[i][0].item()- quantiles[i][1].item(),3) for i in range(3)]) #subtract to get the difference
    return tuple(aux)

mu_norm = sum(df.normal)/len(df)
var_norm = sum(df.normal**2)/len(df) - (sum(df.normal)/len(df))**2

def q2():
    interval_left = df.normal.mean()-df.normal.std()
    interval_right = df.normal.mean()+ df.normal.std()

    prob_left = sct.norm.cdf(interval_left, loc=20, scale=4) 
    prob_ = sct.norm.cdf(interval_right, loc=20, scale=4) 
    prob = prob_ - prob_left
    return prob.item()

q2()

def q3():
    m_binom = df.binomial.mean().item()
    v_binom = df.binomial.var().item()
    m_norm = df.normal.mean().item()
    v_norm = df.normal.var().item()
    return (round(m_binom - m_norm,3), round(v_binom - v_norm,3))

stars = pd.read_csv("data/pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)

stars.describe()

stars[stars['target']==True]

stars[stars['target']==True].describe()

false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
sns.distplot(false_pulsar_mean_profile)
plt.show()

false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - false_pulsar_mean_profile.mean())/false_pulsar_mean_profile.std()
sns.distplot(false_pulsar_mean_profile_standardized, color = 'orange', label= 'Standardized Non Pulsar stars')
#sns.distplot(sct.norm.rvs(0, 1, size=10000), color ='blue' , label='Theoric normal')
plt.show()

q1_norm, q2_norm, q3_norm = [sct.norm.ppf(x, loc = 0, scale = 1) for x in [0.80, 0.90, 0.95]]
q1_norm, q2_norm, q3_norm # valores X para os determinados valores da PDF  [0.80, 0.90, 0.95]

def q4():
    false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
    false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - false_pulsar_mean_profile.mean())/false_pulsar_mean_profile.std()
    q1_norm, q2_norm, q3_norm = [sct.norm.ppf(x, loc = 0, scale = 1) for x in [0.80, 0.90, 0.95]]
    q1 = round((false_pulsar_mean_profile_standardized<=q1_norm).sum()/len(false_pulsar_mean_profile_standardized),3).item()
    q2 = round((false_pulsar_mean_profile_standardized<=q2_norm).sum()/len(false_pulsar_mean_profile_standardized),3).item()
    q3 = round((false_pulsar_mean_profile_standardized<=q3_norm).sum()/len(false_pulsar_mean_profile_standardized),3).item()
    return (q1, q2, q3) 

def q5():
    false_pulsar_mean_profile = stars.loc[stars['target']==False, 'mean_profile']
    false_pulsar_mean_profile_standardized = (false_pulsar_mean_profile - false_pulsar_mean_profile.mean())/false_pulsar_mean_profile.std()
    q1_norm, q2_norm, q3_norm = [sct.norm.ppf(x, loc = 0, scale = 1).item() for x in [0.25, 0.5, 0.75]]
    q1_pulsar, q2_pulsar, q3_pulsar = [false_pulsar_mean_profile_standardized.quantile(q = x).item() for x in [0.25, 0.50, 0.75]] #get quantiles of the dataframe 
    diff1 = round(q1_pulsar - q1_norm,3)
    diff2 = round(q2_pulsar - q2_norm,3)
    diff3 = round(q3_pulsar - q3_norm,3)
    return (diff1,diff2,diff3)

(q5())
