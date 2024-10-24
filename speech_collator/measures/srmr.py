from scipy.signal.windows import hamming
from srmrpy.hilbert import hilbert
from srmrpy.modulation_filters import *
from gammatone.fftweight import fft_gtgram
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
from srmrpy.segmentaxis import segment_axis
from srmrpy.srmr import calc_erbs, calc_cutoffs, normalize_energy

from scipy.io.wavfile import read as readwav

def srmr(x, fs, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=False, norm=True):
    # Computing gammatone envelopes
    if fast:
        mfs = 400.0
        gt_env = fft_gtgram(x, fs, 0.010, 0.0025, n_cochlear_filters, low_freq)
    else:
        cfs = centre_freqs(fs, n_cochlear_filters, low_freq)
        fcoefs = make_erb_filters(fs, cfs)
        gt_env = np.abs(hilbert(erb_filterbank(x, fcoefs)))
        mfs = fs
    wLength = 1024
    wInc = 256
    # Computing modulation filterbank with Q = 2 and 8 channels
    mod_filter_cfs = compute_modulation_cfs(min_cf, max_cf, 8)
    MF = modulation_filterbank(mod_filter_cfs, mfs, 2)
    n_frames = int(1 + (gt_env.shape[1] - wLength)//wInc)
    w = hamming(wLength+1)[:-1] # window is periodic, not symmetric
    energy = np.zeros((n_cochlear_filters, 8, n_frames))
    for i, ac_ch in enumerate(gt_env):
        mod_out = modfilt(MF, ac_ch)
        for j, mod_ch in enumerate(mod_out):
            mod_out_frame = segment_axis(mod_ch, wLength, overlap=wLength-wInc, end='pad')
            energy[i,j,:] = np.sum((w*mod_out_frame[:n_frames])**2, axis=1)
    if norm:
        energy = normalize_energy(energy)
    erbs = np.flipud(calc_erbs(low_freq, fs, n_cochlear_filters))
    avg_energy = np.mean(energy, axis=2)
    total_energy = np.sum(avg_energy)
    AC_energy = np.sum(avg_energy, axis=1)
    AC_perc = AC_energy*100/total_energy
    AC_perc_cumsum=np.cumsum(np.flipud(AC_perc))
    K90perc_idx = np.where(AC_perc_cumsum>90)[0][0]
    BW = erbs[K90perc_idx]
    cutoffs = calc_cutoffs(mod_filter_cfs, fs, 2)[0]
    if (BW > cutoffs[4]) and (BW < cutoffs[5]):
        Kstar=5
    elif (BW > cutoffs[5]) and (BW < cutoffs[6]):
        Kstar=6
    elif (BW > cutoffs[6]) and (BW < cutoffs[7]):
        Kstar=7
    elif (BW > cutoffs[7]):
        Kstar=8
    return np.sum(avg_energy[:, :4])/np.sum(avg_energy[:, 4:Kstar]), (np.sum(energy[:, :4], axis=0).sum(axis=0)/np.sum(energy[:, 4:Kstar], axis=0).sum(axis=0)) 
