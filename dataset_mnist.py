from matplotlib.pyplot import show
from dataset import *


class MnistDataset(Dataset):
    def __init__(self):
        super(MnistDataset, self).__init__('mnist', 'select')
        
        tr_x_path = '../Dataset/MNIST/train-images-idx3-ubyte'
        tr_y_path = '../Dataset/MNIST/train-labels-idx1-ubyte'
        
        xs = np.fromfile(tr_x_path, dtype='uint8')[16:]
        ys = np.fromfile(tr_y_path, dtype='uint8')[8:]
        
        xs = xs.reshape([-1, 28*28])
        ys = np.eye(10)[ys]
        
        self.shuffle_data(xs, ys)
        
        
def mnist_visualize(self, xs, estimates, answers):
    dump_text(answers, estimates)
    dump_image_data(xs)
MnistDataset.visualize = mnist_visualize
    
    
def dump_text(answers, estimates):
    ans = np.argmax(answers, axis=1)
    est = np.argmax(estimates, axis=1)
    print('정답', ans, 'vs. ','추정', est)
    print()
    
def dump_image_data(images):

    show_cnt = len(images)
    fig, axes = plt.subplots(1, show_cnt, figsize =(show_cnt,1))
    
    for n in range(show_cnt):
        plt.subplot(1, show_cnt, n+1)
        plt.imshow(images[n].reshape(28, 28), cmap='Greys_r')
        plt.axis('off')
    
    
    plt.draw()
    plt.show()
    print("---------------------------------------")
    print()
