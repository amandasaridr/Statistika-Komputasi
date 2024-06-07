#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import scipy.stats as stats
import random

# Menghasilkan 100 pengamatan acak dengan rentang nilai 11 hingga 20
def jacknife():
  x = [random.randint(11, 20) for y in range(100)]
  mean_lengths, n = [], len(x)
  index = np.arange(n)

  for z in range(n):
      jk_sample = np.delete(x, z)
      mean_lengths.append(np.mean(jk_sample))

  mean_lengths = np.array(mean_lengths)
  jk_mean_lengths = np.mean(mean_lengths)

  jk_var = (n-1)*np.var(mean_lengths)
  jk_lower_ci = jk_mean_lengths - 1.96 * np.sqrt(jk_var)
  jk_upper_ci = jk_mean_lengths + 1.96 * np.sqrt(jk_var)

  print('jk_mean adalah', jk_mean_lengths)
  print("Jacknife 95% CI lower = {}, upper = {}".format(jk_lower_ci, jk_upper_ci))

jacknife()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.special import gamma

def pdf_gamma(x, alpha, beta):
  return(gamma(alpha+beta) / (gamma(alpha) * gamma(beta))) * x**(alpha-1) * (1-x) * (1-x)**(beta-1)

def pdf_proposal(x):
  return stats.uniform.pdf(x)

def inverse_cdf_gamma(u, alpha, beta):
  return stats.gamma.ppf(u, alpha, beta)

alpha = 1.5
beta = 1
n = 500

u = np.random.uniform(size=n)
rand_data = inverse_cdf_gamma(u, alpha, beta)

print(rand_data)


plt.figure(figsize=(8, 4))
sns.set_style("ticks")

sns.histplot(data=rand_data, stat='density', color='blue', alpha=0.6)

x = np.linspace(0, 1, 500)
y = pdf_gamma(x, alpha, beta)
sns.lineplot(x=x, y=y, color='black')

plt.show()


# In[9]:


import numpy as np
import scipy.stats as stats
import random

# Menghasilkan 100 pengamatan acak dengan rentang nilai 11 hingga 20
def jacknife():
  x = [random.randint(11, 20) for _ in range(100)]
  mean_lengths, n = [], len(x)
  index = np.arange(n)

  for i in range(n):
      jk_sample = np.delete(x, i)
      mean_lengths.append(np.mean(jk_sample))

  mean_lengths = np.array(mean_lengths)
  jk_mean_lengths = np.mean(mean_lengths)

  jk_var = (n-1)*np.var(mean_lengths)
  jk_lower_ci = jk_mean_lengths - 1.96 * np.sqrt(jk_var)
  jk_upper_ci = jk_mean_lengths + 1.96 * np.sqrt(jk_var)

  print('jk_mean adalah', jk_mean_lengths)
  print("Jacknife 95% CI lower = {}, upper = {}".format(jk_lower_ci, jk_upper_ci))

jacknife()


# In[11]:


import numpy as np

def median_bootstrap(data, iterations, confidence_level):
    n = len(data)
    medians = []

    for y in range(iterations):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        median = np.median(bootstrap_sample)
        medians.append(median)

    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    lower_bound = np.percentile(medians, lower_percentile * 100)
    upper_bound = np.percentile(medians, upper_percentile * 100)

    return lower_bound, upper_bound

# Simulasi data
np.random.seed(123)  # Untuk hasil yang dapat direproduksi
sample_data = np.random.randint(1, 11, size=50)

# Parameter
bootstrap_iterations = 500
confidence_level = 0.95

# Menjalankan algoritma median bootstrap
lower_bound, upper_bound = median_bootstrap(sample_data, bootstrap_iterations, confidence_level)

# Menampilkan hasil
print("Selang kepercayaan 95% untuk median:")
print("Batas bawah:", lower_bound)
print("Batas atas:", upper_bound)


# In[ ]:




