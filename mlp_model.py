from mathutil import *

np.random.seed(1234)

def randomize(): np.random.seed(time())

class Model(object):
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = dataset
        self.is_training = False
        if not hasattr(self, 'rand_std'): self.rand_std = 0.003
        
        
    def __str__(self):
        return '{}/{}'.format(self.name, self.dataset)

    def exec_all(self, epoch_count=10, batch_size=10, learning_rate = 0.001,\
        report=0, show_cnt=3):
        
        self.train(epoch_count, batch_size, learning_rate, report)      #학습하기
        self.test()                                                     #Test하기
        if show_cnt > 0 : self.visualize(show_cnt)
        

class MlpModel(Model):
    def __init__(self, name, dataset, hconfigs):            # 학습모델의 이름, Dataset의 객체, [12,3]  ==> hidden의 크기
        super(MlpModel, self).__init__(name, dataset)       #Model에서 name이랑 dataset은 가져옴.
        self.init_parameters(hconfigs)                      ### hconfigs = hyperparameter 임.
        
#====================== (init-parameters) =================================
#여러 class에 가져다 쓰기 위헤서 아래와 같이 Def 따로 정의함.
def mlp_init_parameters(self, hconfigs):
    self.hconfigs = hconfigs
    self.pm_hiddens =[]
    
    prev_shape = self.dataset.input_shape           
    # Q)
    
    #↓ pm_hidden{'w', 'b'} 의 initiate 
    for hconfig in hconfigs:
        pm_hidden, prev_shape = self.alloc_layer_param(prev_shape, hconfig)  #다음 layer가 느끼는 prev_shape 저장해두기
        self.pm_hiddens.append(pm_hidden)
        
    output_cnt = int(np.prod(self.dataset.output_shape))
    # Q)
    self.pm_output, _ = self.alloc_layer_param(prev_shape, output_cnt)   # output이 마지막이라서 prev_shape가 필요 없음.
MlpModel.init_parameters = mlp_init_parameters

    
def mlp_alloc_layer_param(self, input_shape, hconfig):
    input_cnt = np.prod(input_shape)
    output_cnt = hconfig
    weight, bias = self.alloc_param_pair([input_cnt, output_cnt])
    return {'w':weight, 'b':bias}, output_cnt   #pm_hidden , 다음에겐 이게 prev_shape(prev_cnt)
MlpModel.alloc_layer_param = mlp_alloc_layer_param

def mlp_alloc_param_pair(self, shape): # ex shape =[input_cnt, output_cnt]
    weight = np.random.normal(0, self.rand_std, shape)
    bias =  np.zeros(shape[-1])
    return weight, bias
MlpModel.alloc_param_pair = mlp_alloc_param_pair

#================================================================================================    
    
def mlp_model_train(self, epoch_count=10, batch_size =10,\
    learning_rate = 0.001, report =0):
    self.learning_rate = learning_rate
    
    batch_count = int(self.dataset.train_count / batch_size)        # 천제 갯수 / batch size
    time1 = time2 = int(time.time())
    if report != 0:
        print('Model {} train started:'.format(self.name))
        
    for epoch in range(epoch_count):
        costs = []
        accs = []
        self.dataset.shuffle_train_data(batch_size * batch_count)   # 나머지 버림
        for n in range(batch_count):
            trX, trY = self.dataset.get_train_data(batch_size, n)   # batch size만큼씩 nth를 가져옴
            cost, acc = self.train_step(trX, trY)                   #부분 학습
            costs.append(cost)
            accs.append(acc)
            
        if report >0 and (epoch+1) % report ==0:
            vaX, vaY = self.dataset.get_validate_data(100)          #새로 받아온 100개의 Data로 validation을 시작함
            acc = self.eval_accuracy(vaX, vaY)
            time3 = int(time.time())
            tm1, tm2 = time3-time2, time3-time1
            self.dataset.train_prt_result(epoch+1, costs, accs, acc, tm1, tm2)
            time2 = time3
            
    tm_total = int(time.time())-time1
    print('Model {} train ended in {} secs :'.format(self.name, tm_total))
MlpModel.train = mlp_model_train

def mlp_model_test(self):   #Data받아서 accuracy 돌리기
    teX, teY = self.dataset.get_test_data()
    time1 = int(time.time())
    acc = self.eval_accuracy(teX, teY)
    time2 = int(time.time())
    self.dataset.test_prt_result(self.name, acc, time2-time1)
MlpModel.test = mlp_model_test

def mlp_model_visualize(self, num):     # Q)num은 show_cnt 예시로 보여줌
    print('Model {} Visualization'.format(self.name))            
    deX, deY = self.dataset.get_visualize_data(num)
    est = self.get_estimate(deX)
    self.dataset.visualize(deX, est, deY)       
    # dataset의 visualize를 사용하는 이유는 출력내용을 맞춤형으로 하기 위해서임.
    # visual data 가져와서 - estimation 하고 나서 그것을 비교하여 나타냄
MlpModel.visualize = mlp_model_visualize

def mlp_train_step(self, x, y):     # x: input, y: label
    self.is_train = True
    
    output, aux_nn = self.forward_neuralnet(x)          # Q) hconfig data는 어디서 들어갈까? self.hconfigs 에 들어가 있음.
    loss, aux_pp = self.forward_postproc(output, y)
    accuracy = self.eval_accuracy(x, y, output)
    
    G_loss = 1.0
    G_output = self.backprop_postproc(G_loss, aux_pp)   # aux_pp 는 사실상 output
    self.backprop_neuralnet(G_output, aux_nn)
    
    self.is_training = False
    return loss, accuracy
MlpModel.train_step = mlp_train_step
    
def mlp_forward_neuralnet(self, x):
    hidden = x
    aux_layers = []
    
    for n, hconfig in enumerate(self.hconfigs):
        hidden, aux = self.forward_layer(hidden, hconfig, self.pm_hiddens[n])   
        #(pre, hidden수: relu 유무 판단, W) ==> output, pre
        #forward_layer : return y, [x, y]
        aux_layers.append(aux)
        
    output, aux_out = self.forward_layer(hidden, None, self.pm_output)
    return output, [aux_out, aux_layers]    #output, aux_out은 [hidden_last, output=y=x*w+b], aux_
MlpModel.forward_neuralnet=mlp_forward_neuralnet


def mlp_forward_layer(self, x, hconfig, pm):
    y = np.matmul(x, pm['w'])+pm['b']
    if hconfig is not None: y=relu(y)
    return y, [x, y]
MlpModel.forward_layer = mlp_forward_layer
        

def mlp_backprop_neuralnet(self, G_output, aux):
    aux_out, aux_layers = aux     
    
    G_hidden = self.backprop_layer(G_output, None, self.pm_output, aux_out)
    #첫번째 G_output부터 거꾸로 오는 방향
    
    for n in reversed(range(len(self.hconfigs))):
        hconfig, pm, aux = self.hconfigs[n], self.pm_hiddens[n], aux_layers[n]
        G_hidden = self.backprop_layer(G_hidden, hconfig, pm, aux)      # dL/dx를 뱉어냄.
    
    return G_hidden
MlpModel.backprop_neuralnet= mlp_backprop_neuralnet        


def mlp_backprop_layer(self, G_y, hconfig, pm, aux):
    x, y = aux
    if hconfig is not None: G_y = relu_derv(y)*G_y
    
    g_y_weight = x.transpose()
    g_y_input = pm['w'].transpose()
    
    G_weight = np.matmul(g_y_weight, G_y)
    G_bias = np.sum(G_y, axis=0)
    G_input = np.matmul(G_y, g_y_input)
    
    pm['w'] -= self.learning_rate * G_weight
    pm['b'] -= self.learning_rate * G_bias
    
    return G_input
    
MlpModel.backprop_layer = mlp_backprop_layer       
    
 

   
def mlp_forward_postproc(self, output, y):
    loss, aux_loss = self.dataset.forward_postproc(output, y) #forward_postproc --> dataset.forward_postproc : mode는 dataset에서 입력
    extra, aux_extra = self.forward_extra_cost(y)    
    return loss + extra, [aux_loss, aux_extra] # 여기서는 extra, aux_extra 이건 정규화 regulation의 추가 손실 분이다.
MlpModel.forward_postproc = mlp_forward_postproc

def mlp_forward_extra_cost(self, y):
    return 0, None
MlpModel.forward_extra_cost = mlp_forward_extra_cost

def mlp_backprop_postproc(self, G_loss, aux):
    aux_loss, aux_extra = aux
    self.backprop_extra_cost(G_loss, aux_extra)
    G_output = self.dataset.backprop_postproc(G_loss, aux_loss)
    return G_output
MlpModel.backprop_postproc=mlp_backprop_postproc

def mlp_backprop_extra_cost(self, G_loss, aux):
    pass

MlpModel.backprop_extra_cost = mlp_backprop_extra_cost

def mlp_eval_accuracy(self, x, y, output=None): #output은 forward_neuralnet의 결과물로 반복연산을 피하려는 목적임
    if output is None:
        output, _ = self.forward_neuralnet(x)
    accuracy = self.dataset.eval_accuracy(x, y, output)
    return accuracy
MlpModel.eval_accuracy = mlp_eval_accuracy

def mlp_get_estimate(self, x):
    output, _ = self.forward_neuralnet(x)
    estimate = self.dataset.get_estimate(output)
    return estimate
MlpModel.get_estimate = mlp_get_estimate

#% forward(backprop)_postproc,  eval_accuracy, get_estimate : dataset에서 제공하는 함수를 이용한다.

    

        
        
        
        