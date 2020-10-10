import numpy as np
import torch
from net import UNet
import cv2
from dataset import SEGData

class FeatureExtractor():
    """ 提取对应层的特征 """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            # =========数据在该模块的前向过程=============#
            x = module(x)
            if name in self.target_layers:
                # 将目标特征层的反传时的梯度hook出来
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """
    重新定义前向过程，得到：
    1.网络的输出；
    2.特定的中间层的激活；
    3.特定的中间层的梯度。
    """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        # ===这里不同网路不同设计====
        for i, (name, module) in enumerate(self.model._modules.items()):
            if module == self.feature_module:
                print('extract')
                target_activations, x = self.feature_extractor(x)

            # elif "avgpool" in name.lower():
            #     # print('2' * 20)
            #     x ,x_= module(x)
            #     x = x.view(x.size(0), -1)
            # UNet的结构，和前向一样
            else:
                print(i)
                print(name)
                print(module)
                if i == 0:
                    x_1, x = module(x)
                if i == 1:
                    x_2, x = module(x)
                if i == 2:
                    x_3, x = module(x)
                if i == 3:
                    x_4, x = module(x)
                if i == 4:
                    x = module(x, x_4)
                if i == 5:
                    x = module(x, x_3)
                if i == 6:
                    x = module(x, x_2)
                if i == 7:
                    x = module(x, x_1)

        return target_activations, x


class GradCam():
    def __init__(self, model, feature_module, targe_layer_names):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.model = model.cuda()
        self.extractor = ModelOutputs(self.model, self.feature_module, targe_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        features, output = self.extractor(input.cuda())
        # 得到每个每个像素点位置的每个通道上的最大值
        index = np.argmax(output.cpu().data.numpy(), axis=1)
        one_hot = np.zeros((1, output.size()[-2], output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # 这里使用one-hot做mask取output值
        value = torch.sum(one_hot.cuda() * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        value.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # 特征层的梯度
        target = features[-1]  # 网络种指定特征层的输出
        target=target.cpu().data.numpy()[0,:]
        """
         计算 
        $$alpha_k ^c= \frac{1}{Z}\sum_i\sum_j\frac{\partial y^c}{\partial A^k_{ij}}$$
        """
        alpha = np.mean(grads_val, axis=(2, 3))[0,:]
        """
        计算
        $$L^c _{Grad-CAM}=ReLU(\sum_k \alpha _k^c A^k)$$
        """
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(alpha):
            cam += w * target[i, :, :] #sum
        cam=np.maximum(cam,0)#relu

        #由于特征图太小，还得缩放到原图大小
        cam=cv2.resize(cam,input.shape[2:])
        #
        cam=cam-np.min(cam)
        cam=cam/np.max(cam)

        cam_ = np.array(cam * 255, dtype=np.uint8)
        cam_ = cv2.applyColorMap(cam_, cv2.COLORMAP_JET)
        # cv2.imshow('grad', im_color)
        # cv2.waitKey()
        cv2.imwrite('cam.jpg',cam_)

        return cam

def show_cam_on_imgage(img,mask):
    heatmap=cv2.applyColorMap(np.uint8(255*mask),cv2.COLORMAP_JET)
    heatmap=np.float32(heatmap)/255#归一化
    cam=heatmap+np.float32(img)
    cam=cam/np.max(cam)
    cv2.imwrite('cam_img.jpg',np.uint8(255*cam))

if __name__ == '__main__':
    model = UNet().cuda()
    model.load_state_dict(torch.load('SAVE/Unet.pt'))
    print(model.o)
    grad_cam=GradCam(model=model,feature_module=model.o,
                     targe_layer_names=['5'])
    data=SEGData()
    img,lable=data[90]
    input=torch.unsqueeze(img,dim=0).cuda()

    mask=grad_cam(input)
    img_hwc=img.permute(1,2,0)
    show_cam_on_imgage(img_hwc,mask)
