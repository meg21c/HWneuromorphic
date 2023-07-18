#%%
from adam_model import *
from hw_neuro_model import *
from dataset_office32 import *
from dataset_mnist import *


# %%
# om1 = MlpModel('office31_model_1',od,[10])
# om1.exec_all(epoch_count=20, report=10)
# %%
md = MnistDataset()
params=[[0.0617, -1.10,  10.14, 0.0 ],[ 0.1464, 0.659, 10.148,  0 ]]
mm = HwModel('Mnist', md,[100],params)
mm.exec_all(epoch_count=100,batch_size=400, report=2,learning_rate=0.0001)

# %%

# od = Office31Dataset()

# om2=AdamModel('office31_model_2',od,[64,32,10])
# om2.exec_all(epoch_count=50, report=10, learning_rate=0.0001)

# om3=AdamModel('office31_model_3',od,[64,32,10])
# om3.use_adam = True
# om3.exec_all(epoch_count=50, report=10, learning_rate=0.0001)