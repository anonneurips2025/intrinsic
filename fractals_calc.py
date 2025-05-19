#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kernels import * 
from empirical_width import *  
from generate_sample import *


# In[4]:


def kernel_delta_from_euclidean(euclead_delta, kernel_function):
    euclead_delta = np.array([euclead_delta])
    zero = np.array([0])
    kernel_delta = np.sqrt(2-2*kernel_function(euclead_delta,zero,diagonal=2))
    return kernel_delta


# In[5]:


sample, euclead_delta = generate_cantor_sample(15)
widths_squared = calc_sequence(sample, laplace_kernel, nSize=300)


# In[6]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

laplace_delta = kernel_delta_from_euclidean(euclead_delta, laplace_kernel)

Y = -0.5*np.log(np.array(widths_squared[20:]))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(21,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(2*np.log(2)/np.log(3))

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[7]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})
sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+laplace_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("Laplace kernel on Cantor set")
plt.savefig('laplace_cantor.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[8]:


sample, euclead_delta = generate_cantor_sample(17)
widths_squared = calc_sequence(sample, one_over_two_kernel, nSize=300)


# In[9]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

one_over_two_delta = kernel_delta_from_euclidean(euclead_delta, one_over_two_kernel)

Y = -0.5*np.log(np.array(widths_squared[0:]))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(4*np.log(2)/np.log(3))

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[10]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})
sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+one_over_two_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("$e^{-\sqrt{\|x-y\|}}$ on Cantor set")
plt.savefig('one_over_two_cantor.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[11]:


sample, euclead_delta = generate_weierstrass_sample(24)
widths_squared = calc_sequence(sample, laplace_kernel, 200)


# In[12]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

laplace_delta = kernel_delta_from_euclidean(euclead_delta, laplace_kernel)

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(3.0)

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[13]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})
sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+laplace_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("Laplace kernel on Weierstrass function graph")
plt.savefig('laplace_weierstrass.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[14]:


sample, euclead_delta = generate_weierstrass_sample(24)
widths_squared = calc_sequence(sample, one_over_two_kernel, 300)


# In[15]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

one_over_two_delta = kernel_delta_from_euclidean(euclead_delta, one_over_two_kernel)

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(6.0)

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[16]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})
sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+one_over_two_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("$e^{-\sqrt{\|x-y\|}}$ on Weierstrass function graph")
plt.savefig('one_over_two_weierstrass.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[17]:


sample, euclead_delta = generate_sierpinski_sample(8) 
widths_squared = calc_sequence(sample, laplace_kernel, 300)


# In[18]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

laplace_delta = kernel_delta_from_euclidean(euclead_delta, laplace_kernel)

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(2*np.log(8)/np.log(3))

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[19]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})
sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+laplace_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("Laplace kernel on Sierpinski carpet")
plt.savefig('laplace_sierpinski.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[20]:


sample, euclead_delta = generate_sierpinski_sample(8) 
widths_squared = calc_sequence(sample, one_over_two_kernel, 300)


# In[21]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

one_over_two_delta = kernel_delta_from_euclidean(euclead_delta, one_over_two_kernel)

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(4*np.log(8)/np.log(3))

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[22]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})
sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+one_over_two_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("$e^{-\sqrt{\|x-y\|}}$ on Sierpinski carpet")
plt.savefig('one_over_two_sierpinski.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[24]:


sample = generate_lorenz_sample(1000000)/50
widths_squared = calc_sequence(sample, laplace_kernel)


# In[25]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(2*2.06)

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[26]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})

sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed)
ax = plt.gca()
ax.set_title("Laplace kernel on Lorenz attractor")
plt.savefig('laplace_lorenz.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[27]:


sample, euclead_delta = generate_menger_sample(5)
widths_squared = calc_sequence(sample, laplace_kernel, 300)


# In[28]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

laplace_delta = kernel_delta_from_euclidean(euclead_delta, laplace_kernel)

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(2*np.log(20)/np.log(3))

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[29]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})

sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+laplace_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("Laplace kernel on Menger sponge")
plt.savefig('laplace_menger.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[30]:


sample, euclead_delta = generate_menger_sample(5)
widths_squared = calc_sequence(sample, one_over_two_kernel, 300)


# In[31]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

one_over_two_delta = kernel_delta_from_euclidean(euclead_delta, one_over_two_kernel)

Y = -0.5*np.log(np.array(widths_squared))
N=len(widths_squared)
X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
print(1/est2.params[1])
print(2*np.log(20)/np.log(3))

ransac = linear_model.RANSACRegressor()
ransac.fit(X2, Y)
print(1/ransac.estimator_.coef_[1])


# In[32]:


needed = pd.DataFrame({'$\log(n)$': np.log(np.arange(1,N+1)), '$-\log(w_K(n))$' : -0.5*np.log(np.array(widths_squared))})

sns.lmplot(x="$\log(n)$", y="$-\log(w_K(n))$", data=needed, scatter=None)
ax = plt.gca()
x=np.log(np.arange(1,N+1))
y=-np.log(np.sqrt(np.array(widths_squared)))
err2 = np.array([0.0]*N)
err1=np.log(np.sqrt(np.array(widths_squared))+one_over_two_delta)-np.log(np.sqrt(np.array(widths_squared)))
err = np.stack((err1,err2))
plt.errorbar(x, y, yerr=err, fmt="o", markersize=0.3, elinewidth=0.3, color='k')

ax.set_title("$e^{-\sqrt{\|x-y\|}}$ on Menger sponge")
plt.savefig('one_over_two_menger.pdf', dpi=400, bbox_inches='tight')
plt.show()



