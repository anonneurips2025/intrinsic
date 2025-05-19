from kernels import * 
from empirical_width import *  
from generate_sample import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import special
import math
import statsmodels.api as sm
from sklearn import linear_model

def calc_base_a (dim, kernel_function):
    deltas = []
    for euclead_delta in np.arange(0.01,0.11,0.01):
        deltas.append(kernel_delta_from_euclidean_on_sphere(euclead_delta, dim, kernel_function))
    X = np.log(np.arange(0.01,0.11,0.01))
    Y = np.log(np.array(deltas))
    X2 = sm.add_constant(X)
    est = sm.OLS(Y, X2)
    est2 = est.fit()
    #print(est2.summary())
    print(est2.params[1])

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X2, Y)
    a = ransac.estimator_.coef_[1]
    return a


# In[31]:


print("For ReLU")
calc_base_a(3,nngp1_relu)
print("For exp")
calc_base_a(3,nngp1_exp)
print("For sigmoid")
calc_base_a(3,nngp1_sigmoid)
print("For tanh")
calc_base_a(3,nngp1_tanh)
print("For erf")
calc_base_a(3,nngp1_erf)
print("For NTK ReLU")
calc_base_a(3,ntk1_relu)
print("For NTK exp")
calc_base_a(3,ntk1_exp)
print("For NTK sigmoid")
calc_base_a(3,ntk1_sigmoid)
print("For NTK tanh")
calc_base_a(3,ntk1_tanh)
print("For NTK erf")
calc_base_a(3,ntk1_erf)
print("For ReLU")
calc_base_a(3,nngp2_relu)
print("For exp")
calc_base_a(3,nngp2_exp)
print("For sigmoid")
calc_base_a(3,nngp2_sigmoid)
print("For tanh")
calc_base_a(3,nngp2_tanh)
print("For erf")
calc_base_a(3,nngp2_erf)
print("For NTK ReLU")
calc_base_a(3,ntk2_relu)
print("For NTK exp")
calc_base_a(3,ntk2_exp)
print("For NTK sigmoid")
calc_base_a(3,ntk2_sigmoid)
print("For NTK tanh")
calc_base_a(3,ntk2_tanh)
print("For NTK erf")
calc_base_a(3,ntk2_erf)


# In[19]:


import numpy as np
import statsmodels.api as sm
from sklearn import linear_model

def calc_kolm_dims(kernel_function):
    kolm_dims = []
    for dim in range(3,7):
        sample = generate_sphere_sample(50,dim)
        widths_squared = calc_sequence(sample, kernel_function, 5)
        widths_squared = widths_squared[0:]
        Y = -0.5*np.log(np.array(widths_squared))
        N=len(widths_squared)
        X = np.transpose(np.array([np.log(np.arange(1,N+1))]))
        X2 = sm.add_constant(X)
        ransac = linear_model.RANSACRegressor()
        ransac.fit(X2, Y)
        d_K = 1/ransac.estimator_.coef_[1]
        if d_K<0:
            d_K=0
        kolm_dims.append(d_K)
    return kolm_dims


# In[32]:


kolm_dims = calc_kolm_dims(nngp1_relu)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(1)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 1-Layer ReLU kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp1_relu_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp1_erf)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(0)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 1-Layer erf kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp1_erf_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp1_exp)
plt.figure(2)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 1-Layer $e^{-x^2/2}$ kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp1_exp_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp1_sigmoid)
plt.figure(3)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 1-Layer sigmoid kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp1_sigmoid_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp1_tanh)
plt.figure(4)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 1-Layer tanh kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp1_tanh_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[20]:


kolm_dims = calc_kolm_dims(ntk1_relu)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(4,12,2), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(5)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 1-Layer ReLU kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk1_relu_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk1_erf)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(6)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 1-Layer erf kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk1_erf_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk1_exp)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(7)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 1-Layer $e^{-x^2/2}$ kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk1_exp_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk1_sigmoid)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(8)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 1-Layer sigmoid kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk1_sigmoid_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk1_tanh)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(9)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 1-Layer tanh kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk1_tanh_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp2_relu)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(10)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 2-Layer ReLU kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp2_relu_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp2_erf)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(11)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 2-Layer erf kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp2_erf_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp2_exp)
plt.figure(12)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 2-Layer $e^{-x^2/2}$ kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp2_exp_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp2_sigmoid)
plt.figure(13)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 2-Layer sigmoid kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp2_sigmoid_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(nngp2_tanh)
plt.figure(14)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NNGP 2-Layer tanh kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('nngp2_tanh_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk2_relu)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(4,12,2), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(15)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 2-Layer ReLU kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk2_relu_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk2_erf)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(16)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 2-Layer erf kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk2_erf_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk2_exp)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(17)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 2-Layer $e^{-x^2/2}$ kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk2_exp_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk2_sigmoid)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(18)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 2-Layer sigmoid kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk2_sigmoid_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()


# In[ ]:


kolm_dims = calc_kolm_dims(ntk2_tanh)
needed = pd.DataFrame({'$d$': range(3,7), '$d^{\rm emp}_ϱ$' : range(2,6), '$d^{\rm emp}_K$' : np.array(kolm_dims)})
plt.figure(19)
ax = plt.gca()
a = plt.scatter(x='$d$', y='$d^{\rm emp}_ϱ$', data=needed, label="$d^{\rm emp}_ϱ$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_ϱ$'])
b = plt.scatter(x='$d$', y='$d^{\rm emp}_K$', data=needed, label="$d^{\rm emp}_K$", s=10)
plt.plot(needed['$d$'], needed['$d^{\rm emp}_K$'])
plt.legend((a, b),("$d^{emp}_ϱ$", "$d^{emp}_K$"),scatterpoints=1,loc='upper left',fontsize=8)

ax.set_title("NTK 2-Layer tanh kernel on $S^{d-1}$")
ax.set_xlabel('ambient dimension')
ax.set_ylim(bottom=0)
plt.savefig('ntk2_tanh_sphere.pdf', dpi=400, bbox_inches='tight')
plt.show()

