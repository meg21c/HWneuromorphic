# HW neuromorhpic을 위한 updating 을 하려고 함.
# 1) min max 설정이 필요하고
# 2) 기울기 대신 up/donw 에 대한 평가가 필요함

from ctypes import sizeof
from mlp_model import  *

class HwModel(MlpModel):
   def __init__(self, name, dataset, hconfigs, params):
       self.use_hw = False
       self.params = params
       super(HwModel,self).__init__(name, dataset, hconfigs)
       #params=[['Ap', 'Bp', 'Gpmax','Gpmin'], ['Ad', 'Bd', 'Gdmax', 'Gdmin']]
       
#use_adam이 True 인지 False인지에 따라서 update방식을 변경하기 위해서 만든 method
def Hw_backprop_layer(self, G_y, hconfig, pm, aux):
    x, y = aux       
    
    if hconfig is not None: G_y =relu_derv(y)*G_y
    
    g_y_weight = x.transpose()
    G_weight = np.matmul(g_y_weight, G_y)

    G_bias = np.sum(G_y, axis =0)
    
    g_y_input = pm['w'].transpose()
    G_input = np.matmul(G_y, g_y_input)
    
    self.update_param(pm, 'w', G_weight) #depend on "use_adam" True or False
    self.update_param(pm, 'b', G_bias)
     
    return G_input

HwModel.backprop_layer = Hw_backprop_layer


def Hw_update_param(self, pm, key, delta):
    if self.use_hw:
        delta = self.eval_hw_delta(pm, key, delta)
        
    pm[key] -= self.learning_rate *delta        # x2 = x1 - a* [m^/sqrt(v^+e)]=delta
HwModel.update_param = Hw_update_param


def Hw_eval_hw_delta(self, pm, key, delta):
    #---------------------------------------------------
    
    G_weight_flag_posi=np.where(delta>0,-self.dLTD(pm[key]),0)
    G_weight_flag_nega=np.where(delta<0,self.dLTP(pm[key]),0)
    
    G_weight = G_weight_flag_posi+G_weight_flag_nega    
    #--------------------------------------------------- 여기까지

    
    return G_weight
HwModel.eval_hw_delta = Hw_eval_hw_delta


def dLTP(self, curW):
    Ap, Bp, Gpmax, Gpmin = self.params[0]
    LTP=Ap*np.exp(-Bp*((curW-Gpmin)/(Gpmax-Gpmin)))
    LTP=np.where(LTP>Gpmax, Gpmax, LTP)
    LTP=np.where(LTP<Gpmin, Gpmin, LTP)
    return LTP
HwModel.dLTP = dLTP


def dLTD(self, curW):
    Ad, Bd, Gdmax, Gdmin = self.params[1]
    LTD=Ad*np.exp(-Bd*((Gdmax-curW)/(Gdmax-Gdmin)))
    LTD=np.where(LTD>Gdmax, Gdmax, LTD)
    LTD=np.where(LTD<Gdmin, Gdmin, LTD)
    return LTD
HwModel.dLTD = dLTD


# def Hw_update_param(self, pm, key, delta):
#     if self.use_adam:
#         delta = self.eval_adam_delta(pm, key, delta)
    
#     pm[key] -= self.learning_rate *delta        # x2 = x1 - a* [m^/sqrt(v^+e)]=delta
# HwModel.update_param = Hw_update_param

# def adam_eval_adam_delta(self, pm, key, delta):
#     ro_1 = 0.9
#     ro_2 = 0.999
#     epsilon = 1e-8
#     skey, tkey, step = 's'+key, 't'+key, 'n'+key        #'w' or 'b' = key, sw, tw, nw
#     if skey not in pm:
#         pm[skey] = np.zeros(pm[key].shape)
#         pm[tkey] = np.zeros(pm[key].shape)
#         pm[step]=0
        
#     s = pm[skey] = ro_1*pm[skey] +(1-ro_1)*delta
#     t = pm[tkey] = ro_2*pm[tkey] +(1-ro_2)*(delta*delta)
    
#     pm[step] +=1
#     s = s / (1-np.power(ro_1, pm[step]))
#     t = t / (1-np.power(ro_2, pm[step]))
    
#     return s / (np.sqrt(t)+epsilon)
# HwModel.eval_adam_delta = adam_eval_adam_delta
    

    
        
        