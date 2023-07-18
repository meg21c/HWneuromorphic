from mlp_model import  *

class AdamModel(MlpModel):
   def __init__(self, name, dataset, hconfigs):
       self.use_adam = False
       super(AdamModel,self).__init__(name, dataset, hconfigs)
       
       
#use_adam이 True 인지 False인지에 따라서 update방식을 변경하기 위해서 만든 method
def adam_backprop_layer(self, G_y, hconfig, pm, aux):
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

AdamModel.backprop_layer = adam_backprop_layer

def adam_update_param(self, pm, key, delta):
    if self.use_adam:
        delta = self.eval_adam_delta(pm, key, delta)
    
    pm[key] -= self.learning_rate *delta        # x2 = x1 - a* [m^/sqrt(v^+e)]=delta
AdamModel.update_param = adam_update_param

def adam_eval_adam_delta(self, pm, key, delta):
    ro_1 = 0.9
    ro_2 = 0.999
    epsilon = 1e-8
    skey, tkey, step = 's'+key, 't'+key, 'n'+key        #'w' or 'b' = key, sw, tw, nw
    if skey not in pm:
        pm[skey] = np.zeros(pm[key].shape)
        pm[tkey] = np.zeros(pm[key].shape)
        pm[step]=0
        
    s = pm[skey] = ro_1*pm[skey] +(1-ro_1)*delta
    t = pm[tkey] = ro_2*pm[tkey] +(1-ro_2)*(delta*delta)
    
    pm[step] +=1
    s = s / (1-np.power(ro_1, pm[step]))
    t = t / (1-np.power(ro_2, pm[step]))
    
    return s / (np.sqrt(t)+epsilon)



AdamModel.eval_adam_delta = adam_eval_adam_delta
    

    
        
        