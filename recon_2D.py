import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch

import matplotlib.pyplot as plt

def phantom_2d(device):
    obj=torch.ones((128,128),device=device)
    obj[16:48,16:48]=0.2
    obj[16:48,80:112]=0.4
    obj[80:112,16:48]=0.6
    obj[80:112,80:112]=0.8
    return obj

def siddon(src,det,N,vox_size,device="cuda"):
    NLine, NDIM=src.shape
    ray=det-src

    ray_length=torch.norm(ray,dim=1)
    t_list=[]
    for i in range(NDIM):
        i_cor=torch.arange(N[i]+1,device=device)*vox_size[i]-0.5*vox_size[i]*N[i]
        seg=i_cor.reshape((1,-1))-src[:,i].reshape((NLine,1))
        iray=ray[:,i]
        iray=torch.where(iray==0,torch.tensor([1.0e-10],device=device),iray)
        iray=iray[:,None].expand((NLine, N[i]+1))
        t=seg/iray
        # t=torch.where((t<0) | (t>1), torch.tensor([0],device=device),t)
        t_list.append(t)
    t_all=torch.stack(t_list).permute(1,2,0).reshape((NLine,-1))
    t_all=torch.sort(t_all,axis=1)[0]

    t_mid=(t_all[:,1:]+t_all[:,:-1])*0.5
    p_mid=src.unsqueeze(1)+t_mid.unsqueeze(2)*ray.unsqueeze(1)
    low_limit=(-0.5*vox_size*N)
    # low_limit=low_limit.reshape((1,1,NDIM)).repeat(p_mid.shape[0],p_mid.shape[1],1)
    idx=torch.floor((p_mid-low_limit)/vox_size).long()
    # print(idx.shape)

    cond0=(idx[:,:,0]>=0) & (idx[:,:,0]<N[0])
    cond1=(idx[:,:,1]>=0) & (idx[:,:,1]<N[1])
    mask=cond0&cond1
    if NDIM==3:
        cond2= (idx[:,:,2]>=0) & (idx[:,:,2]<N[2])
        mask &= cond2

    seg_len = t_all[:, 1:] - t_all[:, :-1]

    line_indices = torch.arange(NLine, device=device).unsqueeze(1).expand(mask.shape)
    # print(line_indices.shape, idx.shape, seg_len.shape, mask.shape)
    valid_lines = line_indices[mask]
    valid_idx = idx[mask]
    valid_len = seg_len[mask] * ray_length[valid_lines]

    if NDIM==2:
        weight=torch.zeros((NLine,N[1],N[0]),device=device)
        weight[valid_lines, valid_idx[:, 1], valid_idx[:, 0]] = valid_len
        return weight
    elif NDIM==3:
        weight = torch.zeros((NLine, N[2], N[1], N[0]), device=device)
        weight[valid_lines, valid_idx[:,2], valid_idx[:,1], valid_idx[:,0]] = valid_len
        return weight

def radon(obj,src,det,angle,vox_size,device):
    n_angle=angle.shape[0]
    n_det=det.shape[0]
    prj = torch.zeros((n_angle, n_det), device=device)
    for iang in range(n_angle):
        ang = angle[iang]

        c = torch.cos(ang).squeeze()
        s = torch.sin(ang).squeeze()

        rotMat = torch.tensor([
            [c, -s],
            [s, c]
        ], device=device, dtype=torch.float32)

        src_rot = src @ rotMat.T
        det_rot = det @ rotMat.T

        weight = siddon(src_rot, det_rot, torch.tensor([obj.shape[0], obj.shape[1]], dtype=torch.int32, device=device),
                        vox_size, device)

        iprj = weight * obj.unsqueeze(0)
        # print(iprj.shape)
        prj[iang, :] = iprj.sum(dim=(1, 2))
    return prj
def design_filter(filter,length,d):
    order=0
    while 1:
        if np.pow(2,order)>=2*length:
            break
        order+=1
    filter_len=np.power(2,order)
    new_len=filter_len//2+1
    n=np.arange(new_len)
    filtImpResp=np.zeros(new_len)
    filtImpResp[0]=0.25
    filtImpResp[1::2]=-1./((np.pi*n[1::2])**2)
    filtImpResp=np.hstack((filtImpResp,filtImpResp[:0:-1]))
    filt=2.*np.real(np.fft.fft(filtImpResp))
    filt=filt[:new_len]
    w=2*np.pi*(np.arange(filt.shape[0]))/filter_len

    if filter=="ram-lak":
        pass
    elif filter=="shepp-logan":
        filt[1:]=filt[1:]*np.sin(w[1:]/(2*d))/(w[1:]/(2*d))
    elif filter=="cosin":
        filt[1:]=filt[1:]*np.cos(w[1:]/(2*d))
    elif filter=="hamming":
        filt[1:]=filt[1:]*(0.54+0.46*np.cos(w[1:]/d))
    elif filter=="hann":
        filt[1:] = filt[1:] * (1.+np.cos(w[1:]/d))/2.

    filt[w>np.pi*d]=0
    filt=np.hstack((filt,filt[-2:0:-1]))
    # plt.plot(filt)
    return filt

def recon_2D_equal_space(data,SOD,SDD,delta_angle,delta_pix,n_recon,delta_recon,device):
    eps = 1e-8
    n_angle, n_det=data.shape
    filter=design_filter("shepp-logan",n_det,delta_pix)
    filter=torch.from_numpy(filter).to(device)
    # filter = torch.fft.fftshift(filter)
    # prj=torch.from_numpy(data).to(device)
    prj=data

    det_pos = (torch.arange(n_det, device=device) - (n_det / 2 - 0.5)) * delta_pix
    weighted_cos = SDD / torch.sqrt(SDD**2 + det_pos**2 + eps)
    prj_filted1 = prj * weighted_cos.unsqueeze(0)

    prj_padded=torch.nn.functional.pad(prj_filted1,(0,filter.shape[0]-n_det,0,0))
    prj_filted_fft=torch.fft.fft(prj_padded,axis=1)*filter.unsqueeze(0)
    prj_filted_ifft=torch.real(torch.fft.ifft(prj_filted_fft,axis=1))
    prj_filted=prj_filted_ifft[:,:n_det]
    # print(prj_filted.shape)

    n_batch=100
    angle_pos = torch.arange(n_angle, device=device) * delta_angle
    recon = torch.zeros((n_angle, n_recon, n_recon), device=device)
    for idx_ang in range(0,n_angle,n_batch):
        end = min(idx_ang + n_batch, n_angle)
        iang=angle_pos[idx_ang:end]
        iprj=prj_filted[idx_ang:end, :]
        costheta = torch.cos(iang).unsqueeze(1)
        sintheta = torch.sin(iang).unsqueeze(1)
        tmp_x = (torch.arange(n_recon, device=device) - (n_recon / 2 - 0.5)) * delta_recon
        tmp_y = tmp_x
        x,y=torch.meshgrid(tmp_x,tmp_y,indexing='ij')
        x=torch.reshape(x,(-1,)).unsqueeze(0)
        y=torch.reshape(y,(-1,)).unsqueeze(0)
        x_=costheta*x+sintheta*y
        y_=-sintheta*x+costheta*y

        U=(SOD-y_)/SOD
        ix=x_*SDD/(SOD-y_+eps)
        idx=torch.searchsorted(det_pos,ix)
        idx=torch.clamp(idx,1,n_det-1)
        x0=det_pos[idx-1]
        x1=det_pos[idx]
        y0=iprj.gather(1,idx-1)
        y1=iprj.gather(1,idx)

        iy = y0 + (ix - x0) * (y1 - y0) / (x1 - x0+ eps)
        # print(x0.shape,y0.shape,iy.shape)

        iv=torch.reshape(iy/(U**2+eps),(-1,n_recon,n_recon))
        recon[idx_ang:end,:,:]=iv

    recon_=torch.sum(recon,dim=0)
    # plt.imshow(recon_.cpu())
    return recon_

def backprojection(prj,weight,eps):
    tmp=prj/(torch.sum(weight,dim=(1,2))+eps)
    return weight*tmp.unsqueeze(1).unsqueeze(2)

def forwardprojection(recon,weight,eps):
    tmp=recon.unsqueeze(0)*weight
    return torch.sum(tmp,dim=(1,2))

def ART_2D(type,lambda_,n_iter, data,src,det,delta_angle,delta_pix,n_recon,delta_recon,device):
    ### type==1: SART, type==2: SIRT
    eps = 1e-8
    n_angle, n_det = data.shape
    prj = data
    angle_pos = torch.arange(n_angle, device=device) * delta_angle

    recon=torch.zeros([n_recon,n_recon], dtype=torch.float, device=device)
    # backprojection_numerator=torch.zeros([n_recon,n_recon],dtype=torch.float,device=device)
    # backprojection_denominator=torch.zeros([n_recon,n_recon],dtype=torch.float,device=device)
    # lambda_=0.5
    for iloop in range(n_iter):
        backprojection_numerator = torch.zeros([n_recon, n_recon], dtype=torch.float, device=device)
        backprojection_denominator = torch.zeros([n_recon, n_recon], dtype=torch.float, device=device)
        if type==1:
            print(f"SART -- loop {iloop}")
        elif type==2:
            print(f"SIRT -- loop {iloop}")
        for idx_angle in range(n_angle):
            ang = angle_pos[idx_angle]

            c = torch.cos(ang).squeeze()
            s = torch.sin(ang).squeeze()

            rotMat = torch.tensor([
                [c, -s],
                [s, c]
            ], device=device, dtype=torch.float32)

            src_rot = src @ rotMat.T
            det_rot = det @ rotMat.T

            weight = siddon(src_rot, det_rot, torch.tensor([n_recon,n_recon], dtype=torch.int32, device=device),
                            torch.tensor([delta_recon,delta_recon], device=device), device)
            # print(torch.isnan(weight).any().item())
            tmp=recon.unsqueeze(0)*weight
            ei=prj[idx_angle,:]-torch.sum(tmp,dim=(1,2))
            numerator=torch.sum(backprojection(ei,weight,eps),dim=0)
            backprojection_numerator +=numerator

            denominator=torch.sum(weight, dim=0)+eps
            backprojection_denominator +=denominator
            if type==1:
                recon=recon+lambda_*numerator/denominator
        if type==2:
            recon = recon + lambda_ * backprojection_numerator / backprojection_denominator

        # print(ei.shape, sum_m_Aim.shape, sum_ray_Aim.shape,numerator.shape,denominator.shape)
    return recon

def CGLS_2D_fb(type,input_prj,input_recon,n_batch, eps, src,det,delta_angle,delta_pix,n_recon,delta_recon, device):
    n_angle, n_det = input_prj.shape
    angle_pos = torch.arange(n_angle, device=device) * delta_angle

    angle_pos = torch.arange(n_angle, device=device) * delta_angle
    prj=torch.zeros_like(input_prj)
    obj=torch.zeros_like(input_recon)
    for idx_ang in range(0, n_angle, n_batch):
        end = min(idx_ang + n_batch, n_angle)
        ang = angle_pos[idx_ang:end]

        c = torch.cos(ang).view(-1, 1)
        s = torch.sin(ang).view(-1, 1)

        rotMat = torch.stack([
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1)
        ], dim=-2).squeeze(1)

        src_rot = src.unsqueeze(0) @ rotMat
        det_rot = det.unsqueeze(0) @ rotMat

        src_rot = src_rot.view(-1, 2)
        det_rot = det_rot.view(-1, 2)
        # print(src_rot.shape)

        weight = siddon(src_rot, det_rot, torch.tensor([n_recon, n_recon], dtype=torch.int32, device=device),
                        torch.tensor([delta_recon, delta_recon], device=device), device)
        if type==0:
            iprj = forwardprojection(input_recon, weight, eps)
            prj[idx_ang:end, :]=iprj.view(-1,n_det)
            del weight, iprj, src_rot, det_rot
            torch.cuda.empty_cache()
        elif type==1:
            # print(input_prj[idx_ang:end,:].view(-1,1).unsqueeze(-1).shape, weight.shape)
            iobj=input_prj[idx_ang:end,:].reshape(-1, 1, 1)*weight
            obj+=torch.sum(iobj,dim=0)
            del weight, iobj, src_rot, det_rot
            torch.cuda.empty_cache()
    if type==0:
        return prj
    elif type==1:
        return obj


def CGLS_2D(n_iter, data,src,det,delta_angle,delta_pix,n_recon,delta_recon,device):
    eps = 1e-8
    n_angle, n_det = data.shape
    n_batch = 10

    x=torch.zeros([n_recon, n_recon], dtype=torch.float, device=device)

    def A(img):
        return CGLS_2D_fb(
            0, data, img, n_batch, eps,
            src, det, delta_angle, delta_pix,
            n_recon, delta_recon, device
        )

    def AT(prj):
        return CGLS_2D_fb(
            1, prj, x, n_batch, eps,
            src, det, delta_angle, delta_pix,
            n_recon, delta_recon, device
        )

    r=data-A(x)
    s=AT(r)
    p=s.clone()
    gamma = torch.sum(s * s)
    for iloop in range(n_iter):
        print(f"CGLS -- loop {iloop}: ", end='')
        q = A(p)
        alpha = gamma / (torch.sum(q * q) + eps)
        x = x + alpha * p
        r = r - alpha * q
        s_new = AT(r)
        gamma_new = torch.sum(s_new * s_new)

        beta = gamma_new / (gamma + eps)

        p = s_new + beta * p
        s = s_new
        gamma = gamma_new
        print(
            "residual norm:",
            torch.sqrt(torch.sum(r * r)).item(),
            "normal residual norm:",
            torch.sqrt(gamma).item()
        )

    return x

def MLEM_2D(n_iter, data,src,det,delta_angle,delta_pix,n_recon,delta_recon,device):
    eps = 1e-8
    n_angle, n_det = data.shape
    n_batch = 10
    x = torch.ones([n_recon, n_recon], dtype=torch.float, device=device)
    def A(img):
        return CGLS_2D_fb(
            0, data, img, n_batch, eps,
            src, det, delta_angle, delta_pix,
            n_recon, delta_recon, device
        )

    def AT(prj):
        return CGLS_2D_fb(
            1, prj, x, n_batch, eps,
            src, det, delta_angle, delta_pix,
            n_recon, delta_recon, device
        )

    img = torch.ones([n_recon, n_recon], dtype=torch.float, device=device)
    ones_sino=torch.ones_like(data)
    bk_ones=AT(ones_sino)
    bk_ones[bk_ones < 1e-6] = 1e-6
    print(ones_sino.shape, bk_ones.shape)

    for _ in range(n_iter):
        proj_est=A(img)
        ratio=data/proj_est.clip(min=1.0e-6)
        corr=AT(ratio)
        img = img * corr / bk_ones

    return img

if __name__=="__main__":
    pass

