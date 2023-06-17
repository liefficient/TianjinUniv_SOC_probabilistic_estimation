# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:56:06 2022

@author: LiGuanzheng
"""

#****************************************NGBoost:25C;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC****************************************

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


from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor

import properscoring as prscore
import pickle
from pathlib import Path
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

from ngboost.distns import Exponential, Normal,LogNormal
from ngboost.scores import LogScore, CRPScore,MLE

tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=4,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_depth=6,
    splitter="best",
    random_state=88,
)   
ngb = NGBRegressor(
                   Base=tree_learner, 
                   n_estimators=1500)
ngb.fit(df_train_X.values,df_train_Y.values[1:].ravel())

#US06_80SOC BJDST_80SOC DST_80SOC
df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

y_pred_test= ngb.predict(df_test_new)
y_dists_test = ngb.pred_dist(df_test_new)

mean = y_dists_test.loc
std = y_dists_test.scale

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



#****************************************NGBoost:0C;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC****************************************

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


from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor

import properscoring as prscore
import pickle
from pathlib import Path
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

random.seed(888)
np.random.seed(888)

# number 10171
BJDST_80SOC=np.load('BJDST_80SOC_0.npy')
BJDST_80SOC=pd.DataFrame(BJDST_80SOC)
BJDST_80SOC.columns=['V','I','SOC']

# number 9526
DST_80SOC=np.load('DST_80SOC_0.npy')
DST_80SOC=pd.DataFrame(DST_80SOC)
DST_80SOC.columns=['V','I','SOC']

# number 9706
FUDS_80SOC=np.load('FUDS_80SOC_0.npy')
FUDS_80SOC=pd.DataFrame(FUDS_80SOC)
FUDS_80SOC.columns=['V','I','SOC']

# number 9481
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

from ngboost.distns import Exponential, Normal,LogNormal
from ngboost.scores import LogScore, CRPScore,MLE

tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=4,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_depth=6,
    splitter="best",
    random_state=88,
)   
ngb = NGBRegressor(
                   Base=tree_learner, 
                   n_estimators=1500)
ngb.fit(df_train_X.values,df_train_Y.values[1:].ravel())

#US06_80SOC BJDST_80SOC DST_80SOC

df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

y_pred_test= ngb.predict(df_test_new)
y_dists_test = ngb.pred_dist(df_test_new)

mean = y_dists_test.loc
std = y_dists_test.scale

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


#****************************************NGBoost:45C;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC****************************************

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


from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor

import properscoring as prscore
import pickle
from pathlib import Path
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

random.seed(888)
np.random.seed(888)

# number 10171
BJDST_80SOC=np.load('BJDST_80SOC_45.npy')
BJDST_80SOC=pd.DataFrame(BJDST_80SOC)
BJDST_80SOC.columns=['V','I','SOC']

# number 9526
DST_80SOC=np.load('DST_80SOC_45.npy')
DST_80SOC=pd.DataFrame(DST_80SOC)
DST_80SOC.columns=['V','I','SOC']

# number 9706
FUDS_80SOC=np.load('FUDS_80SOC_45.npy')
FUDS_80SOC=pd.DataFrame(FUDS_80SOC)
FUDS_80SOC.columns=['V','I','SOC']

# number 9481
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

from ngboost.distns import Exponential, Normal,LogNormal
from ngboost.scores import LogScore, CRPScore,MLE

tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=4,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_depth=6,
    splitter="best",
    random_state=88,
)   
ngb = NGBRegressor(
                   Base=tree_learner, 
                   n_estimators=1500)
ngb.fit(df_train_X.values,df_train_Y.values[1:].ravel())

#US06_80SOC BJDST_80SOC DST_80SOC

df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

y_pred_test= ngb.predict(df_test_new)
y_dists_test = ngb.pred_dist(df_test_new)

mean = y_dists_test.loc
std = y_dists_test.scale

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


#****************************************NGBoost:25C,Noise;Training:FUDS_80SOC,Testing:US06_80SOC BJDST_80SOC DST_80SOC*******************************************

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


from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor

import properscoring as prscore
import pickle
from pathlib import Path
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

from ngboost.distns import Exponential, Normal,LogNormal
from ngboost.scores import LogScore, CRPScore,MLE

tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=4,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_depth=6,
    splitter="best",
    random_state=88,
)   
ngb = NGBRegressor(
                   Base=tree_learner, 
                   n_estimators=1500)
ngb.fit(df_train_X.values,df_train_Y.values[1:].ravel())

#US06_80SOC BJDST_80SOC DST_80SOC

df_test=DST_80SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

y_pred_test= ngb.predict(df_test_new)
y_dists_test = ngb.pred_dist(df_test_new)

mean = y_dists_test.loc
std = y_dists_test.scale

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


from sklearn.tree import DecisionTreeRegressor
from ngboost import NGBRegressor

import properscoring as prscore
import pickle
from pathlib import Path
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

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

# number 27304
US06_50SOC=np.load('US06_50SOC_zizhi.npy')
US06_50SOC=pd.DataFrame(US06_50SOC)
US06_50SOC.columns=['V','I','SOC']

# number 34106
UDDS_50SOC=np.load('UDDS_50SOC_zizhi.npy')
UDDS_50SOC=pd.DataFrame(UDDS_50SOC)
UDDS_50SOC.columns=['V','I','SOC']


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

from ngboost.distns import Exponential, Normal,LogNormal
from ngboost.scores import LogScore, CRPScore,MLE

tree_learner = DecisionTreeRegressor(
    criterion="friedman_mse",
    min_samples_split=4,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_depth=6,
    splitter="best",
    random_state=88,
)   
ngb = NGBRegressor(
                   Base=tree_learner, 
                   n_estimators=1500)
ngb.fit(df_train_X.values,df_train_Y.values[1:].ravel())

#DST_50SOC NEDC_50SOC US06_50SOC UDDS_50SOC
df_test=NEDC_50SOC
V_test=df_test[['V']]
I_test=df_test[['I']]

df_test_new=pd.concat([V_test,I_test, V_test.shift(1), I_test.shift(1)], axis=1)
df_test_new = df_test_new.dropna()
df_test_new.columns=['V','I','V_1','I_1']

y_pred_test= ngb.predict(df_test_new)
y_dists_test = ngb.pred_dist(df_test_new)

mean = y_dists_test.loc
std = y_dists_test.scale

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


NEDC_df_test=df_test['SOC'][1:]
NEDC_df_test.to_csv('NEDC_df_test.csv')

NEDC_mean_ngboost=y_pred_test
NEDC_mean_ngboost=pd.DataFrame(NEDC_mean_ngboost)
NEDC_mean_ngboost.to_csv('NEDC_mean_ngboost.csv')

NEDC_lower_0_ngboost=lower[0]
NEDC_lower_0_ngboost=pd.DataFrame(NEDC_lower_0_ngboost)
NEDC_lower_0_ngboost.to_csv('NEDC_lower_0_ngboost.csv')

NEDC_lower_1_ngboost=lower[1]
NEDC_lower_1_ngboost=pd.DataFrame(NEDC_lower_1_ngboost)
NEDC_lower_1_ngboost.to_csv('NEDC_lower_1_ngboost.csv')

NEDC_lower_2_ngboost=lower[2]
NEDC_lower_2_ngboost=pd.DataFrame(NEDC_lower_2_ngboost)
NEDC_lower_2_ngboost.to_csv('NEDC_lower_2_ngboost.csv')

NEDC_upper_0_ngboost=upper[0]
NEDC_upper_0_ngboost=pd.DataFrame(NEDC_upper_0_ngboost)
NEDC_upper_0_ngboost.to_csv('NEDC_upper_0_ngboost.csv')

NEDC_upper_1_ngboost=upper[1]
NEDC_upper_1_ngboost=pd.DataFrame(NEDC_upper_1_ngboost)
NEDC_upper_1_ngboost.to_csv('NEDC_upper_1_ngboost.csv')

NEDC_upper_2_ngboost=upper[2]
NEDC_upper_2_ngboost=pd.DataFrame(NEDC_upper_2_ngboost)
NEDC_upper_2_ngboost.to_csv('NEDC_upper_2_ngboost.csv')


df_result=pd.DataFrame(
                        {'V':d_v,
                         'I':d_c,
                         'SOC':SOC
                            }
    )

df_result.to_csv('UDDS_50SOC_zizhi.csv')