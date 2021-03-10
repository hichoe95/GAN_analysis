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



artifact_index = [4,5,7,15,17,18,19,21,22,34,35,42,44,46,47,51,55,58,67,68,70,73,
        83,84,86,88,97,101,102,107,109,114,120,122,127,136,139,145,146,151,154,
        157,160,161,179,183,184,191,202,205,207,210,211,212,222,229,
        238,239,240,241,242,243,245,248,249,261,262,263,265,267,268,277,279,306,
        328,338,342,344,346,347,362,372,373,379,382,
        395,396,404,408,411,412,417,418,421,422,428,429,436,437,451,455,464,467,
        468,486,504,511,521,526,527,535,541,542,
        556,559,560,567,585,592,594,595,605,608,612,614,616,
        626,627,634,642,643,645,650,671,673,676,679,683,694,699,
        711,718,724,737,751,752,755,757,758,761,764,765,768,772,776,777,778,779,
        783,785,789,793,794,798,802,803,806,827,830,833,835,837,844,851,852,853,854,857,
        870,872,878,890,893,894,895,906,914,931,933,
        940,943,946,949,950,956,959,968,972,974,979,980,982,988,989,990,992,1000,1013,
        1014,1015,1021,1039,1040,1045,1060,1062,1068,1072,1078,1085,
        1095,1100,1113,1114,1123,1124,1134,1138,1142,1147,1152,1160,1166,
        1171,1172,1186,1189,1190,1198,1199,1200,1201,1207,1208,1214,1229,
        1252,1257,1263,1265,1270,1276,1290,1291,1305,
        1330,1362,1364,1368,1372,1377,1381,1383,1389,1390,1401,
        1404,1418,1431,1433,1445,1451,1457,1471,1478,1479,1481,
        1482,1484,1485,1487,1496,1497,1502,1505,1512,1518,1530,1532,1533,1540,1548,
        1564,1570,1571,1573,1576,1579,1580,1585,1587,1588,1593,1594,1600,1616,1628,
        1643,1644,1655,1657,1660,1686,1687,1693,1697,1705,1715,
        1731,1737,1745,1748,1758,1761,1776,
        1799,1802,1809,1810,1812,1825,1828,1833,1838,1847,1848,1849,
        1873,1874,1875,1879,1880,1881,1886,1897,1898,1903,1917,1924,1926,1931,1946,
        1953,1954,1955,1957,1964,1976,1979,1980,1981,1983,1987,2015,
        ]



normal_index = np.setdiff1d(np.array(range(0,2028)), artifact_index)



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
