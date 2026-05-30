import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from recon_2D import *
import matplotlib.pyplot as plt

def test_recon():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obj=phantom_2d(device)
    vox_size=torch.tensor([0.5, 0.5],device=device)

    plt.subplot(3,3,1)
    plt.imshow(obj.cpu(), cmap="gray")
    plt.title("image")

    SOD=200
    SDD=400
    n_det=256
    det_size=1.0
    dx=(torch.arange(n_det,device=device)-(n_det/2-0.5))*det_size
    dy=torch.ones_like(dx)*(SDD-SOD)
    det=torch.cat((dx[:,None],dy[:,None]),dim=1)

    src=torch.tensor([0,-SOD],dtype=torch.float32, device=device).reshape((1,-1)).repeat(det.shape[0],1)
    angle=torch.arange(720,device=device)*0.5/180*torch.pi
    prj=radon(obj,src,det,angle,vox_size,device)
    plt.subplot(3,3,2)
    plt.imshow(prj.cpu(), cmap="gray")
    plt.title('projection')

    recon = recon_2D_equal_space(prj, SOD, SDD, -0.5 / 180 * torch.pi, det_size, 256, 0.25, device)
    plt.subplot(3, 3, 3)
    plt.imshow(recon.cpu(), cmap="gray")
    plt.title('FDK')

    recon = ART_2D(1, 0.4, 3, prj, src, det, 0.5 / 180 * torch.pi, det_size, 256, 0.25,
                   device)
    plt.subplot(3, 3, 4)
    plt.imshow(recon.cpu(), cmap="gray")
    plt.title('SART')
    # exit()

    recon = ART_2D(2,0.7, 20, prj,  src, det, 0.5 / 180 * torch.pi, det_size, 256, 0.25,
                   device)
    plt.subplot(3, 3, 5)
    plt.imshow(recon.cpu(), cmap="gray")
    plt.title('SIRT')

    recon = CGLS_2D(5, prj, src, det, -0.5 / 180 * torch.pi, det_size, 256, 0.25,
                   device)
    plt.subplot(3, 3, 6)
    plt.imshow(recon.cpu(), cmap="gray")
    plt.title('CGLS')

    recon = MLEM_2D(5, prj, src, det, -0.5 / 180 * torch.pi, det_size, 256, 0.25,
                   device)
    plt.subplot(3, 3, 7)
    plt.imshow(recon.cpu(), cmap="gray")
    plt.title('MLEM')


if __name__=="__main__":
    test_recon()
    plt.show()