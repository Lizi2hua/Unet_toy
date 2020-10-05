import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from net import UNet
from dataset import SEGData
from cfg import *
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


net = UNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()
data=SEGData()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,num_workers=0,drop_last=True)
summary=SummaryWriter(r'Log')
EPOCH=1000
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    for i,(img,label) in  enumerate(dataloader):
        img=img.cuda()
        label=label.cuda()
        img_out=net(img)
        # exit()
        loss=loss_func(img_out,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        summary.add_scalar('bceloss',loss,i)

    torch.save(net.state_dict(),r'SAVE/Unet.pt')
    img,label=data[90]
    img=torch.unsqueeze(img,dim=0).cuda()
    net.eval()
    out=net(img)
    save_image(out, 'Log_imgs/segimg_ep{}_90th_pic.jpg'.format(epoch,i), nrow=1, scale_each=True)
    print('第{}轮结束'.format(epoch))









