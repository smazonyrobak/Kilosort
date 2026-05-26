import logging

import numpy as np
import torch 
from torch.nn.functional import conv1d, max_pool2d, max_pool1d
from tqdm import tqdm

from kilosort import CCG, state_compat
from kilosort.utils import log_performance

logger = logging.getLogger(__name__)


def prepare_extract(xc, yc, U, nC, position_limit, device=torch.device('cuda')):
    """Identify desired channels based on distances and template norms.
    
    Parameters
    ----------
    xc : np.ndarray
        X-coordinates of contact positions on probe.
    yc : np.ndarray
        Y-coordinates of contact positions on probe.
    U : torch.Tensor
        TODO
    nC : int
        Number of nearest channels to use.
    position_limit : float
        Max distance (in microns) between channels that are used to estimate
        spike positions in `postprocessing.compute_spike_positions`.

    Returns
    -------
    iCC : np.ndarray
        For each channel, indices of nC nearest channels.
    iCC_mask : np.ndarray
        For each channel, a 1 if the channel is within 100um and a 0 otherwise.
        Used to control spike position estimate in post-processing.
    iU : torch.Tensor
        For each template, index of channel with greatest norm.
    Ucc : torch.Tensor
        For each template, spatial PC features corresponding to iCC.
    
    """
    ds = (xc - xc[:, np.newaxis])**2 +  (yc - yc[:, np.newaxis])**2 
    iCC = np.argsort(ds, 0)[:nC]
    iCC = torch.from_numpy(iCC).to(device)
    iCC_mask = np.sort(ds, 0)[:nC]
    iCC_mask = iCC_mask < position_limit**2
    iCC_mask = torch.from_numpy(iCC_mask).to(device)
    iU = torch.argmax((U**2).sum(1), -1)
    Ucc = U[torch.arange(U.shape[0]),:,iCC[:,iU]]

    return iCC, iCC_mask, iU, Ucc


def extract(ops, bfile, U, device=torch.device('cuda'), progress_bar=None,
            local_template_parent_id=None, local_template_window_id=None,
            local_template_valid_start_sample=None,
            local_template_valid_end_sample=None,
            local_template_edge_padding=0,
            state_compat_compute_residual_energy=False):
    nC = ops['settings']['nearest_chans']
    position_limit = ops['settings']['position_limit']
    iCC, iCC_mask, iU, Ucc = prepare_extract(
        ops['xc'], ops['yc'], U, nC, position_limit, device=device
        )
    ops['iCC'] = iCC
    ops['iCC_mask'] = iCC_mask
    ops['iU'] = iU
    nt = ops['nt']
    
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 
    ctc = prepare_matching(ops, U)
    st = np.zeros((10**6, 3), 'float64')
    tF  = torch.zeros((10**6, nC , ops['settings']['n_pcs']))
    local_templates_active = local_template_parent_id is not None
    if local_templates_active:
        local_template_parent_id = np.asarray(local_template_parent_id)
        local_template_window_id = np.asarray(local_template_window_id)
        local_template_valid_start_sample = np.asarray(local_template_valid_start_sample)
        local_template_valid_end_sample = np.asarray(local_template_valid_end_sample)
        spike_local_template_id = np.zeros(10**6, 'int32')
        spike_parent_template_id = np.zeros(10**6, 'int32')
        spike_template_window_id = np.zeros(10**6, 'int32')
        spike_match_score = np.zeros(10**6, 'float32')
        if state_compat_compute_residual_energy:
            spike_residual_energy = np.zeros(10**6, 'float32')
            spike_residual_energy_normed = np.zeros(10**6, 'float32')
    k = 0
    prog = tqdm(
        np.arange(bfile.n_batches, dtype=np.int64),
        miniters=200 if progress_bar else None, 
        mininterval=60 if progress_bar else None
        )
    
    try:
        for ibatch in prog:
            if ibatch % 100 == 0:
                log_performance(logger, 'debug', f'Batch {ibatch}')

            X = bfile.padded_batch_to_torch(ibatch, ops)
            template_time_mask = None
            if local_templates_active:
                batch_start = int(ibatch) * bfile.batch_downsampling * ops['batch_size']
                template_time_mask = state_compat.template_time_mask(
                    local_template_valid_start_sample,
                    local_template_valid_end_sample,
                    batch_start,
                    X.shape[-1],
                    ops,
                    local_template_edge_padding,
                    device
                )
            stt, amps, th_amps, Xres = run_matching(
                ops, X, U, ctc, device=device, template_time_mask=template_time_mask
            )
            xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ ops['wPCA'].T
            xfeat += amps * Ucc[:,stt[:,1]]
            if local_templates_active and state_compat_compute_residual_energy:
                residual_energy, residual_energy_normed = \
                    state_compat.compute_residual_energy(
                        Xres, U, ops['wPCA'], iCC, iU, Ucc, stt, amps, tiwave
                    )

            if ibatch == 0:
                # Can sometimes get negative spike times for first batch since
                # we're aligning to nt0min, not nt//2, but these should be discarded.
                neg_spikes = (stt[:,0] - nt - nt//2 + ops['nt0min']) < 0
                stt = stt[~neg_spikes,:]
                xfeat = xfeat[:,~neg_spikes,:]
                amps = amps[~neg_spikes,:]
                th_amps = th_amps[~neg_spikes,:]
                if local_templates_active and state_compat_compute_residual_energy:
                    residual_energy = residual_energy[~neg_spikes]
                    residual_energy_normed = residual_energy_normed[~neg_spikes]

            nsp = len(stt) 
            if k+nsp>st.shape[0]:                     
                st = np.concatenate((st, np.zeros_like(st)), 0)
                tF  = torch.cat((tF,  torch.zeros_like(tF)), 0)
                if local_templates_active:
                    spike_local_template_id = np.concatenate((
                        spike_local_template_id,
                        np.zeros_like(spike_local_template_id)
                    ), 0)
                    spike_parent_template_id = np.concatenate((
                        spike_parent_template_id,
                        np.zeros_like(spike_parent_template_id)
                    ), 0)
                    spike_template_window_id = np.concatenate((
                        spike_template_window_id,
                        np.zeros_like(spike_template_window_id)
                    ), 0)
                    spike_match_score = np.concatenate((
                        spike_match_score, np.zeros_like(spike_match_score)
                    ), 0)
                    if state_compat_compute_residual_energy:
                        spike_residual_energy = np.concatenate((
                            spike_residual_energy,
                            np.zeros_like(spike_residual_energy)
                        ), 0)
                        spike_residual_energy_normed = np.concatenate((
                            spike_residual_energy_normed,
                            np.zeros_like(spike_residual_energy_normed)
                        ), 0)

            t_shift = ibatch * bfile.batch_downsampling * (ops['batch_size'])
            stt = stt.double()
            st[k:k+nsp,0] = ((stt[:,0]-nt) + t_shift).cpu().numpy() - nt//2 + ops['nt0min']
            st[k:k+nsp,1] = stt[:,1].cpu().numpy()
            st[k:k+nsp,2] = th_amps.cpu().numpy().squeeze()

            tF[k:k+nsp]  = xfeat.transpose(0,1).cpu()
            if local_templates_active:
                local_ids = stt[:,1].cpu().numpy().astype('int32')
                spike_local_template_id[k:k+nsp] = local_ids
                spike_parent_template_id[k:k+nsp] = local_template_parent_id[local_ids]
                spike_template_window_id[k:k+nsp] = local_template_window_id[local_ids]
                spike_match_score[k:k+nsp] = th_amps.cpu().numpy().squeeze()
                if state_compat_compute_residual_energy:
                    spike_residual_energy[k:k+nsp] = residual_energy.cpu().numpy()
                    spike_residual_energy_normed[k:k+nsp] = \
                        residual_energy_normed.cpu().numpy()

            k+= nsp
            
            if progress_bar is not None:
                progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
    except:
        logger.exception(f'Error in template_matching.extract on batch {ibatch}')
        logger.debug(f'X shape: {X.shape}')
        logger.debug(f'stt shape: {stt.shape}')
        raise

    log_performance(logger, 'debug', f'Batch {ibatch}')

    isort = np.argsort(st[:k,0])
    st = st[isort]
    tF = tF[isort]
    if local_templates_active:
        kwargs = {}
        if state_compat_compute_residual_energy:
            kwargs['residual_energy'] = spike_residual_energy[:k][isort]
            kwargs['residual_energy_normed'] = spike_residual_energy_normed[:k][isort]
        state_compat.store_spike_metadata(
            ops,
            spike_local_template_id[:k][isort],
            spike_parent_template_id[:k][isort],
            spike_template_window_id[:k][isort],
            spike_match_score[:k][isort],
            **kwargs
        )

    return st, tF, ops


def align_U(U, ops, device=torch.device('cuda')):
    Uex = torch.einsum('xyz, zt -> xty', U.to(device), ops['wPCA'])
    X = Uex.reshape(-1, ops['Nchan']).T
    X = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=ops['nt']//2)
    Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, ops['nt'])
    imax = torch.argmax(Xmax, 1)

    Unew = Uex.clone() 
    for j in range(ops['nt']):
        ix = imax==j
        Unew[ix] = torch.roll(Unew[ix], ops['nt']//2 - j, -2)
    Unew = torch.einsum('xty, zt -> xzy', Unew, ops['wPCA'])#.transpose(1,2).cpu()
    return Unew, imax


def postprocess_templates(Wall, ops, clu, st, tF, device=torch.device('cuda'),
                          template_parent_id=None,
                          protect_same_parent_templates=False,
                          return_kept_template_idx=False):
    Wall2, _ = align_U(Wall, ops, device=device)
    #Wall3, _= remove_duplicates(ops, Wall2)
    out = merging_function(
        ops, Wall2.transpose(1,2), clu, st, tF,
        0.9, 'mu', check_dt=False, template_parent_id=template_parent_id,
        protect_same_parent_templates=protect_same_parent_templates,
        return_kept_template_idx=return_kept_template_idx, device=device
        )
    if return_kept_template_idx:
        Wall3, _, _, _, _, kept_template_idx = out
    else:
        Wall3, _, _, _, _ = out
    Wall3 = Wall3.transpose(1,2).to(device)
    if return_kept_template_idx:
        return Wall3, kept_template_idx
    return Wall3


def prepare_matching(ops, U):
    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    #mu = (U**2).sum(-1).sum(-1)**.5
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    UtU = torch.einsum('ikl, jml -> ijkm',  U, U)
    ctc = torch.einsum('ijkm, kml -> ijl', UtU, WtW)

    return ctc


def run_matching(ops, X, U, ctc, device=torch.device('cuda'),
                 template_time_mask=None):
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()

    nm = (U**2).sum(-1).sum(-1)
    #mu = nm**.5 
    #U2 = U / mu.unsqueeze(-1).unsqueeze(-1)

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', U, B)

    trange = torch.arange(-nt, nt+1, device=device) 
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 

    st = torch.zeros((100000,2), dtype = torch.int64, device = device)
    amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    th_amps = torch.zeros((100000,1), dtype = torch.float, device = device)
    k = 0

    Xres = X.clone()
    lam = 20

    for t in range(max_peels):
        # Cf = 2 * B - nm.unsqueeze(-1) 
        # Cf is shape (n_units, n_times)
        Cf = torch.relu(B)**2 /nm.unsqueeze(-1)
        #a = 1 + lam
        #b = torch.relu(B) + lam * mu.unsqueeze(-1)
        #Cf = b**2 / a - lam * mu.unsqueeze(-1)**2

        Cf[:, :nt] = 0
        Cf[:, -nt:] = 0
        if template_time_mask is not None:
            Cf = Cf.masked_fill(~template_time_mask, 0)

        Cfmax, imax = torch.max(Cf, 0)
        Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*nt+1), stride=1, padding=(nt))

        #print(Cfmax.shape)
        #import pdb; pdb.set_trace()
        cnd1 = Cmax[0,0] > Th**2
        cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
        xs = torch.nonzero(cnd1 * cnd2)

        
        if len(xs)==0:
            #print('iter %d'%t)
            break

        iX = xs[:,:1]
        iY = imax[iX]

        #isort = torch.sort(iX)

        nsp = len(iX)
        st[k:k+nsp, 0] = iX[:,0]
        st[k:k+nsp, 1] = iY[:,0]
        amps[k:k+nsp] = B[iY,iX] / nm[iY]
        amp = amps[k:k+nsp]
        th_amps[k:k+nsp] = Cmax[0, 0, iX[:,0], None]**.5

        k+= nsp

        #amp = B[iY,iX] 

        n = 2
        for j in range(n):
            Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n,0]], W)
            B[   :, iX[j::n] + trange]  -= amp[j::n] * ctc[:,iY[j::n,0],:]

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    return  st, amps, th_amps, Xres


def merging_function(ops, Wall, clu, st, tF, r_thresh=0.5, mode='ccg',
                     check_dt=True, template_parent_id=None,
                     protect_same_parent_templates=False,
                     return_kept_template_idx=False,
                     device=torch.device('cuda')):
    clu2 = clu.copy()
    if template_parent_id is not None:
        template_parent_id = np.asarray(template_parent_id)

    Ww = Wall.to(device)
    NN = len(Ww)
    if template_parent_id is not None:
        clu_unq = np.arange(NN, dtype=clu2.dtype)
        ns = np.bincount(clu2[clu2 >= 0].astype('int64'), minlength=NN)
    else:
        clu_unq, ns = np.unique(clu2, return_counts = True)

    isort = np.argsort(ns)[::-1]

    is_merged = np.zeros(NN, 'bool')
    is_good = np.zeros(NN,)

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    if mode == 'ccg':
        is_ref, est_contam_rate = CCG.refract(clu, st[:,0]/ops['fs'],
                                              acg_threshold=acg_threshold,
                                              ccg_threshold=ccg_threshold)

    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    t = 0
    nmerge = 0
    while t<NN:
        #if t%100==0:
            #print(t, nmerge)

        kk = clu_unq[isort[t]]

        if ns[kk] == 0:
            t += 1
            continue

        if (mode == 'ccg') and is_ref[kk]==0:
            t += 1
            continue

        if is_merged[kk]:            
            t += 1
            continue

        mu = (Ww**2).sum((1,2), keepdims=True)**.5
        Wnorm = Ww / (1e-6 + mu)

        UtU = torch.einsum('lk, jlm -> jkm',  Wnorm[kk], Wnorm)
        ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)

        cmax, imax = ctc.max(1)
        cmax[kk] = 0

        jsort = np.argsort(cmax.cpu().numpy())[::-1]

        if mode == 'ccg':
            st0 = st[:,0][clu2==kk] / ops['fs']
        
        is_ccg  = 0
        for j in range(NN):
            jj = jsort[j]
            if cmax[jj] < r_thresh:
                break
            if (
                protect_same_parent_templates
                and template_parent_id is not None
                and template_parent_id[kk] == template_parent_id[jj]
            ):
                continue
            # compare with CCG
            if mode == 'ccg':
                st1 = st[:,0][clu2==jj] / ops['fs']
                _, is_ccg, _ = CCG.check_CCG(st0, st1, acg_threshold=acg_threshold,
                                             ccg_threshold=ccg_threshold)        
            else:
                dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj])
                is_ccg = dmu.abs() < 0.2

            if is_ccg:
                is_merged[jj] = 1
                dt = (imax[kk] -imax[jj]).item()
                if dt != 0 and check_dt:
                    # Get spike indices for cluster jj
                    idx = (clu2 == jj)
                    # Update tF and Wall with shifted features
                    tF, Wall = roll_features(W, tF, Ww, idx, jj, dt)
                    # Shift spike times
                    st[idx,0] -= dt
                
                Ww[kk] = ns[kk]/(ns[kk]+ns[jj]) * Ww[kk] + ns[jj]/(ns[kk]+ns[jj]) * Ww[jj]            
                Ww[jj] = 0
                ns[kk] += ns[jj]
                ns[jj] = 0
                clu2[clu2==jj] = kk            

                break

        if is_ccg==0:            
            t +=1    
        else:                
            nmerge+=1
    
    imap = np.cumsum((~is_merged).astype('int32')) - 1
    if imap.size > 0:
        # Otherwise, everything has been merged into a single cluster
        clu2 = imap[clu2]

    kept_template_idx = np.nonzero(~is_merged)[0]
    Ww = Ww[~is_merged]

    if mode == 'ccg':
        is_ref = is_ref[~is_merged]
    else:
        is_ref = None

    sorted_idx = np.argsort(st[:,0])
    st = np.take_along_axis(st, sorted_idx[..., np.newaxis], axis=0)
    clu2 = clu2[sorted_idx]
    tensor_idx = torch.from_numpy(sorted_idx)
    tF = tF[tensor_idx]
    state_compat.sort_spike_metadata(ops, sorted_idx)

    if return_kept_template_idx:
        return Ww.cpu(), clu2, is_ref, st, tF, kept_template_idx
    return Ww.cpu(), clu2, is_ref, st, tF


def roll_features(wPCA, tF, Wall, spike_idx, clust_idx, dt):
    W = wPCA.cpu()
    # Project from PC space back to sample time, shift by dt
    feats = torch.roll(tF[spike_idx] @ W, shifts=dt, dims=2)
    temps = torch.roll(Wall[clust_idx:clust_idx+1] @ wPCA, shifts=dt, dims=2)

    # For values that "rolled over the edge," set equal to next closest bin
    if dt > 0:
        feats[:,:,:dt] = feats[:,:,dt].unsqueeze(-1)
        temps[:,:,:dt] = temps[:,:,dt].unsqueeze(-1)
    elif dt < 0:
        feats[:,:,dt:] = feats[:,:,dt-1].unsqueeze(-1)
        temps[:,:,dt:] = temps[:,:,dt-1].unsqueeze(-1)

    # Project back to PC space and update tF
    tF[spike_idx] = feats @ W.T
    Wall[clust_idx] = temps @ wPCA.T

    return tF, Wall
