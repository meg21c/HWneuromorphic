from matplotlib.pyplot import show
from dataset import *

class FlowersDataset(Dataset):
    pass

def flowers_init(self, resolution=[100,100], input_shape=[-1]):
    super(FlowersDataset, self).__init__('flowers','select')

    path = '../Dataset/flowers'
    self.target_names = list_dir(path)      #꽃이름을 list로 받는다.

    images=[]
    idxs =[]

    for dx, dname in enumerate(self.target_names):
        subpath = path + '/' + dname
        filenames = list_dir(subpath)
        for fname in filenames:
            if fname[-4:] != '.jpg':            #image아닌것 pass
                continue
            imagepath = os.path.join(subpath, fname)
            pixels = load_image_pixels(imagepath, resolution, input_shape)  #image를 resolution에 맞체 한후 inputshape에 맞는 ndarray로 전달
            images.append(pixels)
            idxs.append(dx)

    self.image_shape = resolution + [3]

    xs = np.asarray(images, np.float32)
    ys = onehot(idxs, len(self.target_names))           #종류별로 1, 2,3, ....eye(종류수)로 onehot을 만든다.

    self.shuffle_data(xs, ys, 0.8)

FlowersDataset.__init__ = flowers_init

def flowers_visualize(self, xs, estimates, answers):
    draw_images_horz(xs, self.image_shape)
    show_select_results(estimates, answers, self.target_names)

FlowersDataset.visualize = flowers_visualize
