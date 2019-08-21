import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import random

index=[random.randint(0,145062) for _ in range(8)]
index=list(set(index))

data=pd.read_csv('train_1.csv').fillna(0).values[:,1:]
print(data.shape)
b=np.array(data,dtype=float)
b=np.log1p(b)
fig=plt.figure()
ax=fig.add_subplot(111)
for i in index:
    plot_acf(b[i,:],lags=540,ax=ax,use_vlines=False,fft=True,alpha=None,zero=False)
plt.show()