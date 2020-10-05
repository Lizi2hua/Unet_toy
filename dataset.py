'''使用voc 2012数据集'''
import torch
import torchvision
from cfg import *
from torch.utils.data import  Dataset
from PIL import  Image
class SEGData(Dataset):
    def __init__(self):
        '''
        根据标注文件去取图片
        '''
        self.img_path=IMG_PATH
        self.label_path=SEGLABE_PATH
        self.label_data=os.listdir(self.label_path)
        self.totensor=torchvision.transforms.ToTensor()
        self.resizer=torchvision.transforms.Resize((256,256))
    def __len__(self):
        return len(self.label_data)
    def __getitem__(self, item):
        '''
        由于输出的图片的尺寸不同，我们需要转换为相同大小的图片。首先转换为正方形图片，然后缩放的同样尺度(256*256)。
        否则dataloader会报错。
        '''
        # 取出图片路径
        img_name = os.path.join(self.label_path, self.label_data[item])
        img_name = os.path.split(img_name)
        img_name = img_name[-1]
        img_name = img_name.split('.')
        img_name = img_name[0] + '.jpg'
        img_data = os.path.join(self.img_path, img_name)
        label_data = os.path.join(self.label_path, self.label_data[item])
        # 将图片和标签都转为正方形
        img = Image.open(img_data)
        label = Image.open(label_data)
        w, h = img.size
        # 以最长边为基准，生成全0正方形矩阵
        slide = max(h, w)
        black_img = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_label = torchvision.transforms.ToPILImage()(torch.zeros(3, slide, slide))
        black_img.paste(img, (0, 0, int(w), int(h)))  # patse在图中央和在左上角是一样的
        black_label.paste(label, (0, 0, int(w), int(h)))
        # 变为tensor,转换为统一大小256*256
        img = self.resizer(black_img)
        label = self.resizer(black_label)
        img = self.totensor(img)
        label = self.totensor(label)
        return img,label

# data=SEGData()
# print(data.__len__())
# i,l=data[1]



