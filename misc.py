from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec



query_name = os.listdir('../GANCP/celebA_query/')
query_name.sort(key=lambda x: x[5:-4])
query_name = query_name[1:]
query = np.array([np.load('../GANCP/celebA_query/'+query_name[x]) for x in range(len(query_name))]).reshape((-1,512))

indice = np.array(range(2028))



def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data




def print_image(image):
    image = image.detach().cpu().numpy()
    
    image = adjust_dynamic_range(image, [image.max(), image.min()], [0,1])
    
    plt.imshow(image[0].transpose(1,2,0))
    plt.show()

    
    
def d_outputs(module : torch.nn.Module, name):
    features = []
   
    def fn(_, __, out):
        features.append(out.detach().cpu().numpy())
        
    hook = eval('module.'+name+'.register_forward_hook(fn)')
    
    for i in tqdm(range(0,30000)):
        #1번 부터 시작.
        image_arr = [np.array(Image.open(f'../GANCP/celebA_analysis/data1024x1024/{i+1:05d}.jpg'))]
        image_arr = np.array(image_arr).transpose(0,3,1,2)
        
        input_batch = adjust_dynamic_range(image_arr, [0, 255], [0,1])
        
        with torch.no_grad():
            module(torch.Tensor(input_batch).cuda(0))
    
    hook.remove()

    return features



class find_data_any():
    def __init__(self, Gs_style, D_style, query, query_index, real_images = '../GANCP/celebA_analysis/data1024x1024/', real_features = './features_4x4.npy',  output_layer = 'layer16', random = True):
        self.Gs_style = Gs_style
        self.D_style = D_style
        
        self.random = random
        self.output_layer = output_layer
        
        self.query = query
        self.query_index = query_index
        self.nidx = 0
        
        self.real_images = real_images
        
        self.real_images_feature = np.load(real_features).reshape(30000,-1)
        self.real_images_sign = self.real_images_feature > 0
        self.dimension = self.real_images_sign.shape[-1]
        print("features_4x4.shape : ", self.real_images_sign.shape)
        
        self.index = None
        self.gen_image, self.gen_image_scale = None, None
        self.gen_image_feature = None
        self.gen_image_sign = None
        self.how_many_sign_equal = None
        self.distance = None
        
        self.get_index()
        self.generate_image()
    
    
    def get_index(self,):
        if self.random:
            self.index = np.random.choice(self.query_index)
        else:
            self.index = self.query_index[self.nidx]
            self.nidx += 1
            if self.nidx >= len(self.query_index):
                self.nidx = 0
                print('Index is out of range')
                return
            
    def generate_image(self, ind = -1):
        
            
        with torch.no_grad():
            if ind == -1:
                self.get_index()
                self.gen_image = self.Gs_style(torch.Tensor([self.query[self.index,...]]).cuda(0))
            else:
                self.gen_image = self.Gs_style(torch.Tensor([self.query[ind,...]]).cuda(0))
            self.gen_image = self.gen_image['image'].detach().cpu().numpy()
            print('gen_image.shape : ', self.gen_image.shape)
            self.gen_image_scale = self.adjust_dynamic_range(self.gen_image, [self.gen_image.min(), self.gen_image.max()], [0,1])
            self.gen_image_feature = self.image_feature(self.gen_image_scale).reshape(1,-1)
            self.gen_image_sign = self.gen_image_feature > 0
            self.how_many_sign_equal = self.is_sign_equal()
            self.distance = self.cal_distance()
        
    def adjust_dynamic_range(self, data, drange_in, drange_out):
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return data

    def image_feature(self, x):
        
        gen_feature = []
        def fn(_, __, o):
            gen_feature.append(o.detach().cpu().numpy())
        hook = eval('self.D_style.'+self.output_layer+'.register_forward_hook(fn)')
        with torch.no_grad():
            self.D_style(torch.Tensor(x).cuda(0))
        hook.remove()
        return gen_feature[0]
    
    
    def is_sign_equal(self,):
        how_many_sign_equal = np.array([(~(self.gen_image_sign ^ self.real_images_sign[index])).sum() for index in range(0,30000)])
        return how_many_sign_equal
    
    def cal_distance(self,):
        distance = np.array([np.linalg.norm(self.gen_image_feature[0,...] - self.real_images_feature[i,...], axis = 0) for i in range(30000)])
        return distance
    
    def minmax_scaler(self, x):
        return (x - x.min()) / (x.max() - x.min())
            
    def forward(self, topn, low = False, show = True, dist = False):
        indice = []
        
        indice = np.argsort(self.how_many_sign_equal)[:topn] if low else np.argsort(-self.how_many_sign_equal)[:topn]
        
        indice_dist = []
        
        if dist:
            indice_dist = np.argsort(-self.distance)[:topn] if low else np.argsort(self.distance)[:topn]

        if show:
            self.plot_figure(indice, indice_dist, dist)
        
        # index of image starts from 1 to 30000. 
        return indice + 1, indice_dist + 1
    
    def plot_figure(self, indice, indice_dist, dist = False):
        if dist:
            gs = gridspec.GridSpec(3, indice.shape[0] + 1, wspace = 0.0, hspace = 0.1)
        else:
            gs = gridspec.GridSpec(1, indice.shape[0] + 1, wspace = 0.0, hspace = 0.1)

        plt.figure(figsize = (50,11))
        plt.tight_layout()
        
        plt.subplot(gs[0,0])
        plt.axis('off')
        plt.title('Generated Image', fontsize=20)
        plt.imshow(self.gen_image_scale[0].transpose(1,2,0))
        
        print('indice : ',indice + 1)
        reals = np.array([np.array(Image.open(self.real_images+'{:05d}.jpg'.format(i+1))) for i in indice])
        
        for i in range(len(indice)):
            plt.subplot(gs[0,i+1])
            plt.axis('off')
            plt.title(f'{self.how_many_sign_equal[indice[i]]:d} / {self.dimension:d}',fontsize = 20)
            plt.imshow(reals[i])
        
        if dist:
            reals = np.array([np.array(Image.open(self.real_images+'{:05d}.jpg'.format(i+1))) for i in indice_dist])
            for i in range(len(indice)):
                plt.subplot(gs[1,i+1])
                plt.axis('off')
                plt.title(f'{self.distance[indice_dist[i]]:.3f}', fontsize = 20)
                plt.imshow(reals[i])

        plt.show()
        
    def __call__(self, topn, low = False, show = True, dist = False):
        return self.forward(topn = topn, low = low, show = show, dist = dist)
    
    

def minmax(x):
    if x.max() == x.min():
        return (x-x.min())/(x.max() - x.min() + 1e-7)
    return (x - x.min())/(x.max() - x.min())



def dist_neighbor(feature_layer):
    # feature_layer.shape = (training_iter_num, batch_size, channel, height, width)
    dist = []
    C = feature_layer[0].shape[1]
    for c in range(C):
        dist_each = 0.
        for t in range(1,2000):
            a = minmax(feature_layer[t][0][c])
            b = minmax(feature_layer[t-1][0][c])
            dist_each += np.abs((a - b).mean())

        dist.append(dist_each)
    return dist



def show_features(feature_layer: list, channel_index_list: list, width: int , threshold = 0. , show_index = False, use_intb = True):

    H = channel_index_list
    W = width

    gs = gridspec.GridSpec(len(H), W, wspace = 0.0, hspace = 0.1)

    plt.figure(figsize = (W*2, len(H)*2))
    plt.tight_layout()
    intb = 2000//W if use_intb else 1
    
    for i,h in enumerate(H):
       # plt.title('i')
        for w in range(W):
            plt.subplot(gs[i,w])
            plt.axis('off')
            plt.title(f'{intb*w}')
            plt.imshow(feature_layer[intb*w][0][h])

    plt.show()
