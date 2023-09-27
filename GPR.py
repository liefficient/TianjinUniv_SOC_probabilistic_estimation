# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:42:56 2022

@author: LiGuanzheng
"""

#****************************************GPR:25C;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC****************************************

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy.io as scio
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import properscoring as prscore
import pickle
from pathlib import Path
import os


random.seed(888)
np.random.seed(888)


# number 11204
BJDST_80SOC=np.load('BJDST_80SOC_25.npy')
BJDST_80SOC=pd.DataFrame(BJDST_80SOC)
BJDST_80SOC.columns=['V','I','SOC']

# number 10620
DST_80SOC=np.load('DST_80SOC_25.npy')
DST_80SOC=pd.DataFrame(DST_80SOC)
DST_80SOC.columns=['V','I','SOC']

# number 11091
FUDS_80SOC=np.load('FUDS_80SOC_25.npy')
FUDS_80SOC=pd.DataFrame(FUDS_80SOC)
FUDS_80SOC.columns=['V','I','SOC']

# number 10679
US06_80SOC=np.load('US06_80SOC_25.npy')
US06_80SOC=pd.DataFrame(US06_80SOC)
US06_80SOC.columns=['V','I','SOC']


df_train=FUDS_80SOC

Vk=df_train[['V']]
Ik=df_train[['I']]
SOCk=df_train[['SOC']]

Vk_1=Vk.shift(1)
Ik_1=Ik.shift(1)
SOCk_1=SOCk.shift(1)

df_train_X=pd.concat([Vk,Ik, Vk_1, Ik_1], axis=1)
df_train_X.columns=['V','I','V_1','I_1']

df_train_X = df_train_X.dropna()

df_train_Y = df_train[['SOC']]


import GPy

kernel = GPy.kern.RBF(input_dim=4,ARD=True)
m = GPy.models.GPRegression(df_train_X,df_train_Y[1:],kernel)

print(m)

m.optimize_restarts(num_restarts = 15)

print(m)

print(m.kern.variance)
print(m.kern.lengthscale)


#US06_80SOC BJDST_80SOC DST_80SOC
df_test=US06_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

mean,var= m.predict(df_test_new.values)
std=np.sqrt(var)

mean=mean.ravel()
std=std.ravel()

lower = []
upper = []
for s in range(1,4):
    lower = lower + [mean - s * std]
    upper = upper + [mean + s * std]
    
MAE = mean_absolute_error(df_test['SOC'][1:], mean)
RMSE= mean_squared_error(df_test['SOC'][1:], mean, squared=False)
MAX=sorted(abs(df_test['SOC'][1:]-mean))[-1]
print(MAE)
print(RMSE)
print(MAX)

PICP = np.sum( (df_test['SOC'][1:] >= lower[1]) & (df_test['SOC'][1:]  <= upper[1])  ) / (df_test['SOC'][1:].shape[0] )
PINAW=((upper[1] - lower[1]).mean(axis=0) )/(np.max(df_test['SOC'][1:])-np.min(df_test['SOC'][1:]))
C = prscore.crps_gaussian(df_test['SOC'][1:], mu=mean, sig=std)
CRPS = C.mean()
print(PICP)
print(PINAW)
print(CRPS)


#****************************************GPR:0C;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC****************************************


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy.io as scio
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import properscoring as prscore
import pickle
from pathlib import Path
import os


random.seed(888)
np.random.seed(888)


# number 11204
BJDST_80SOC=np.load('BJDST_80SOC_0.npy')
BJDST_80SOC=pd.DataFrame(BJDST_80SOC)
BJDST_80SOC.columns=['V','I','SOC']

# number 10620
DST_80SOC=np.load('DST_80SOC_0.npy')
DST_80SOC=pd.DataFrame(DST_80SOC)
DST_80SOC.columns=['V','I','SOC']

# number 11091
FUDS_80SOC=np.load('FUDS_80SOC_0.npy')
FUDS_80SOC=pd.DataFrame(FUDS_80SOC)
FUDS_80SOC.columns=['V','I','SOC']

# number 10679
US06_80SOC=np.load('US06_80SOC_0.npy')
US06_80SOC=pd.DataFrame(US06_80SOC)
US06_80SOC.columns=['V','I','SOC']


df_train=FUDS_80SOC

Vk=df_train[['V']]
Ik=df_train[['I']]
SOCk=df_train[['SOC']]

Vk_1=Vk.shift(1)
Ik_1=Ik.shift(1)
SOCk_1=SOCk.shift(1)

df_train_X=pd.concat([Vk,Ik, Vk_1, Ik_1], axis=1)
df_train_X.columns=['V','I','V_1','I_1']

df_train_X = df_train_X.dropna()

df_train_Y = df_train[['SOC']]


import GPy

kernel = GPy.kern.RBF(input_dim=4,ARD=True)
m = GPy.models.GPRegression(df_train_X,df_train_Y[1:],kernel)

print(m)

m.optimize_restarts(num_restarts = 15)

print(m)

print(m.kern.variance)
print(m.kern.lengthscale)


#US06_80SOC BJDST_80SOC DST_80SOC

df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

mean,var= m.predict(df_test_new.values)
std=np.sqrt(var)

mean=mean.ravel()
std=std.ravel()

lower = []
upper = []
for s in range(1,4):
    lower = lower + [mean - s * std]
    upper = upper + [mean + s * std]
    
MAE = mean_absolute_error(df_test['SOC'][1:], mean)
RMSE= mean_squared_error(df_test['SOC'][1:], mean, squared=False)
MAX=sorted(abs(df_test['SOC'][1:]-mean))[-1]
print(MAE)
print(RMSE)
print(MAX)

PICP = np.sum( (df_test['SOC'][1:] >= lower[1]) & (df_test['SOC'][1:]  <= upper[1])  ) / (df_test['SOC'][1:].shape[0] )
PINAW=((upper[1] - lower[1]).mean(axis=0) )/(np.max(df_test['SOC'][1:])-np.min(df_test['SOC'][1:]))
C = prscore.crps_gaussian(df_test['SOC'][1:], mu=mean, sig=std)
CRPS = C.mean()
print(PICP)
print(PINAW)
print(CRPS)


#****************************************GPR:45C;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC****************************************


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy.io as scio
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import properscoring as prscore
import pickle
from pathlib import Path
import os


random.seed(888)
np.random.seed(888)


# number 11204
BJDST_80SOC=np.load('BJDST_80SOC_45.npy')
BJDST_80SOC=pd.DataFrame(BJDST_80SOC)
BJDST_80SOC.columns=['V','I','SOC']

# number 10620
DST_80SOC=np.load('DST_80SOC_45.npy')
DST_80SOC=pd.DataFrame(DST_80SOC)
DST_80SOC.columns=['V','I','SOC']

# number 11091
FUDS_80SOC=np.load('FUDS_80SOC_45.npy')
FUDS_80SOC=pd.DataFrame(FUDS_80SOC)
FUDS_80SOC.columns=['V','I','SOC']

# number 10679
US06_80SOC=np.load('US06_80SOC_45.npy')
US06_80SOC=pd.DataFrame(US06_80SOC)
US06_80SOC.columns=['V','I','SOC']


df_train=FUDS_80SOC

Vk=df_train[['V']]
Ik=df_train[['I']]
SOCk=df_train[['SOC']]

Vk_1=Vk.shift(1)
Ik_1=Ik.shift(1)
SOCk_1=SOCk.shift(1)

df_train_X=pd.concat([Vk,Ik, Vk_1, Ik_1], axis=1)
df_train_X.columns=['V','I','V_1','I_1']

df_train_X = df_train_X.dropna()

df_train_Y = df_train[['SOC']]


import GPy

kernel = GPy.kern.RBF(input_dim=4,ARD=True)
m = GPy.models.GPRegression(df_train_X,df_train_Y[1:],kernel)

print(m)

m.optimize_restarts(num_restarts = 15)

print(m)

print(m.kern.variance)
print(m.kern.lengthscale)


#US06_80SOC BJDST_80SOC DST_80SOC

df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

mean,var= m.predict(df_test_new.values)
std=np.sqrt(var)

mean=mean.ravel()
std=std.ravel()

lower = []
upper = []
for s in range(1,4):
    lower = lower + [mean - s * std]
    upper = upper + [mean + s * std]
    
MAE = mean_absolute_error(df_test['SOC'][1:], mean)
RMSE= mean_squared_error(df_test['SOC'][1:], mean, squared=False)
MAX=sorted(abs(df_test['SOC'][1:]-mean))[-1]
print(MAE)
print(RMSE)
print(MAX)

PICP = np.sum( (df_test['SOC'][1:] >= lower[1]) & (df_test['SOC'][1:]  <= upper[1])  ) / (df_test['SOC'][1:].shape[0] )
PINAW=((upper[1] - lower[1]).mean(axis=0) )/(np.max(df_test['SOC'][1:])-np.min(df_test['SOC'][1:]))
C = prscore.crps_gaussian(df_test['SOC'][1:], mu=mean, sig=std)
CRPS = C.mean()
print(PICP)
print(PINAW)
print(CRPS)


#****************************************GPR:25C,noise;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC*******************************************

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy.io as scio
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import properscoring as prscore
import pickle
from pathlib import Path
import os


random.seed(888)
np.random.seed(888)


# number 11204
BJDST_80SOC=scio.loadmat('BJDST80SOC_noise.mat')
BJDST_80SOC=BJDST_80SOC['BJDST80SOC_noise']
BJDST_80SOC=pd.DataFrame(BJDST_80SOC)
BJDST_80SOC.columns=['V','I','SOC']

# number 10620
DST_80SOC=scio.loadmat('DST80SOC_noise.mat')
DST_80SOC=DST_80SOC['DST80SOC_noise']
DST_80SOC=pd.DataFrame(DST_80SOC)
DST_80SOC.columns=['V','I','SOC']

# number 11091
FUDS_80SOC=scio.loadmat('FUDS80SOC_noise.mat')
FUDS_80SOC=FUDS_80SOC['FUDS80SOC_noise']
FUDS_80SOC=pd.DataFrame(FUDS_80SOC)
FUDS_80SOC.columns=['V','I','SOC']

# number 10679
US06_80SOC=scio.loadmat('US0680SOC_noise.mat')
US06_80SOC=US06_80SOC['US0680SOC_noise']
US06_80SOC=pd.DataFrame(US06_80SOC)
US06_80SOC.columns=['V','I','SOC']


df_train=FUDS_80SOC

Vk=df_train[['V']]
Ik=df_train[['I']]
SOCk=df_train[['SOC']]

Vk_1=Vk.shift(1)
Ik_1=Ik.shift(1)
SOCk_1=SOCk.shift(1)

df_train_X=pd.concat([Vk,Ik, Vk_1, Ik_1], axis=1)
df_train_X.columns=['V','I','V_1','I_1']

df_train_X = df_train_X.dropna()

df_train_Y = df_train[['SOC']]


import GPy

kernel = GPy.kern.RBF(input_dim=4,ARD=True)
m = GPy.models.GPRegression(df_train_X,df_train_Y[1:],kernel)

print(m)

m.optimize_restarts(num_restarts = 15)

print(m)

print(m.kern.variance)
print(m.kern.lengthscale)


#US06_80SOC BJDST_80SOC DST_80SOC

df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

mean,var= m.predict(df_test_new.values)
std=np.sqrt(var)

mean=mean.ravel()
std=std.ravel()

lower = []
upper = []
for s in range(1,4):
    lower = lower + [mean - s * std]
    upper = upper + [mean + s * std]
    
MAE = mean_absolute_error(df_test['SOC'][1:], mean)
RMSE= mean_squared_error(df_test['SOC'][1:], mean, squared=False)
MAX=sorted(abs(df_test['SOC'][1:]-mean))[-1]
print(MAE)
print(RMSE)
print(MAX)

PICP = np.sum( (df_test['SOC'][1:] >= lower[1]) & (df_test['SOC'][1:]  <= upper[1])  ) / (df_test['SOC'][1:].shape[0] )
PINAW=((upper[1] - lower[1]).mean(axis=0) )/(np.max(df_test['SOC'][1:])-np.min(df_test['SOC'][1:]))
C = prscore.crps_gaussian(df_test['SOC'][1:], mu=mean, sig=std)
CRPS = C.mean()
print(PICP)
print(PINAW)
print(CRPS)

#****************************************Self-made experimental dataset,Training:FUDS_50SOC_zizhi,Testing:DST_50SOC_zizhi、NEDC_50SOC_zizhi****************************************

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy.io as scio
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import properscoring as prscore
import pickle
from pathlib import Path
import os


random.seed(888)
np.random.seed(888)


# number 12765
DST_50SOC=np.load('DST_50SOC_zizhi.npy')
DST_50SOC=pd.DataFrame(DST_50SOC)
DST_50SOC.columns=['V','I','SOC']

# number 19553
FUDS_50SOC=np.load('FUDS_50SOC_zizhi.npy')
FUDS_50SOC=pd.DataFrame(FUDS_50SOC)
FUDS_50SOC.columns=['V','I','SOC']

# number 11091
NEDC_50SOC=np.load('NEDC_50SOC_zizhi.npy')
NEDC_50SOC=pd.DataFrame(NEDC_50SOC)
NEDC_50SOC.columns=['V','I','SOC']

df_train=FUDS_50SOC

Vk=df_train[['V']]
Ik=df_train[['I']]
SOCk=df_train[['SOC']]

Vk_1=Vk.shift(1)
Ik_1=Ik.shift(1)
SOCk_1=SOCk.shift(1)

df_train_X=pd.concat([Vk,Ik, Vk_1, Ik_1], axis=1)
df_train_X.columns=['V','I','V_1','I_1']

df_train_X = df_train_X.dropna()

df_train_Y = df_train[['SOC']]


import GPy

kernel = GPy.kern.RBF(input_dim=4,ARD=True)
m = GPy.models.GPRegression(df_train_X,df_train_Y[1:],kernel)

print(m)

m.optimize_restarts(num_restarts = 15)

print(m)

print(m.kern.variance)
print(m.kern.lengthscale)


#DST_50SOC NEDC_50SOC
df_test=NEDC_50SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

mean,var= m.predict(df_test_new.values)
std=np.sqrt(var)

mean=mean.ravel()
std=std.ravel()


lower = []
upper = []
for s in range(1,4):
    lower = lower + [mean - s * std]
    upper = upper + [mean + s * std]
    
MAE = mean_absolute_error(df_test['SOC'][1:], mean)
RMSE= mean_squared_error(df_test['SOC'][1:], mean, squared=False)
MAX=sorted(abs(df_test['SOC'][1:]-mean))[-1]
print(MAE)
print(RMSE)
print(MAX)

PICP = np.sum( (df_test['SOC'][1:] >= lower[1]) & (df_test['SOC'][1:]  <= upper[1])  ) / (df_test['SOC'][1:].shape[0] )
PINAW=((upper[1] - lower[1]).mean(axis=0) )/(np.max(df_test['SOC'][1:])-np.min(df_test['SOC'][1:]))
C = prscore.crps_gaussian(df_test['SOC'][1:], mu=mean, sig=std)
CRPS = C.mean()
print(PICP)
print(PINAW)
print(CRPS)



#****************************************Self-made experimental dataset2,Training:NEDC_50SOC,Testing:DST_50SOC、FUDS_50SOC****************************************


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import scipy.io as scio
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import properscoring as prscore
import pickle
from pathlib import Path
import os


random.seed(888)
np.random.seed(888)


# number 23785
DST_50SOC=pd.read_csv('DST4Ah50-VISOC2.csv')
DST_50SOC=DST_50SOC[['V','I','SOC']]

# number 21769
FUDS_50SOC=pd.read_csv('FUDS4Ah50-VISOC2.csv')
FUDS_50SOC=FUDS_50SOC[['V','I','SOC']]

# number 16443
NEDC_50SOC=pd.read_csv('NEDC4Ah50-VISOC2.csv')
NEDC_50SOC=NEDC_50SOC[['V','I','SOC']]



df_train=NEDC_50SOC

Vk=df_train[['V']]
Ik=df_train[['I']]
SOCk=df_train[['SOC']]

Vk_1=Vk.shift(1)
Ik_1=Ik.shift(1)
SOCk_1=SOCk.shift(1)

df_train_X=pd.concat([Vk,Ik, Vk_1, Ik_1], axis=1)
df_train_X.columns=['V','I','V_1','I_1']

df_train_X = df_train_X.dropna()

df_train_Y = df_train[['SOC']]


import GPy

kernel = GPy.kern.RBF(input_dim=4,ARD=True)
m = GPy.models.GPRegression(df_train_X,df_train_Y[1:],kernel)

print(m)

m.optimize_restarts(num_restarts = 15)

print(m)

print(m.kern.variance)
print(m.kern.lengthscale)


#FUDS_50SOC DST_50SOC 
df_test=FUDS_50SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

mean,var= m.predict(df_test_new.values)
std=np.sqrt(var)

mean=mean.ravel()
std=std.ravel()


lower = []
upper = []
for s in range(1,4):
    lower = lower + [mean - s * std]
    upper = upper + [mean + s * std]
    
MAE = mean_absolute_error(df_test['SOC'][1:], mean)
RMSE= mean_squared_error(df_test['SOC'][1:], mean, squared=False)
MAX=sorted(abs(df_test['SOC'][1:]-mean))[-1]
print(MAE)
print(RMSE)
print(MAX)

PICP = np.sum( (df_test['SOC'][1:] >= lower[1]) & (df_test['SOC'][1:]  <= upper[1])  ) / (df_test['SOC'][1:].shape[0] )
PINAW=((upper[1] - lower[1]).mean(axis=0) )/(np.max(df_test['SOC'][1:])-np.min(df_test['SOC'][1:]))
C = prscore.crps_gaussian(df_test['SOC'][1:], mu=mean, sig=std)
CRPS = C.mean()
print(PICP)
print(PINAW)
print(CRPS)
