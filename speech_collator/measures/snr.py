from librosa.feature import rms
import numpy as np

def _curve_gt0(x, c1=-0.10511163, c2=-14.13272781, d1=0.05551931, d2=5.79780909, loc=1.63692793, scale1=3.31585597, scale2=-2.06433912):
  x = (scale1/(1 + np.exp(-c1*(x-c2)))) + (scale2*np.log(1 + np.exp(-d1*(x-d2))))
  x += loc
  return x

def _curve_lt0(x, loc=0.40890103, a=0.26238549, b=-2.87692077):
  return np.exp(x*a+b)+loc

def approx_gamma_curve(x, bound=1):
  result = np.zeros(x.shape)
  b_s = _curve_lt0(-bound)
  b_e = _curve_gt0(bound)
  b_k = (b_s - b_e)/2
  result[(x>-bound)&(x<bound)] = (b_s + b_e)/2 - b_k * (x[(x>-bound)&(x<bound)] / (np.abs(bound)))
  result[x<=-bound] = _curve_lt0(x[x<=-bound])
  result[x>=bound] = _curve_gt0(x[x>=bound])
  return result

eps = 1e-10
# next 2 lines define a fancy curve derived from a gamma distribution -- see paper
db_vals = np.linspace(-20, 100, 10_000)
g_vals = approx_gamma_curve(db_vals)

def wada_snr(wav):
    # Direct blind estimation of the SNR of a speech signal.
    #
    # Paper on WADA SNR:
    #   http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf
    #
    # This function was adapted from this matlab code:
    #   https://labrosa.ee.columbia.edu/projects/snreval/#9

    # init

    # peak normalize, get magnitude, clip lower bound
    wav = np.array(wav)
    wav[wav==0] = eps
    wav = wav / abs(wav).max()
    abs_wav = abs(wav)

    # calcuate statistics
    # E[|z|]
    v1 = max(eps, abs_wav.mean())
    v1_t = rms(y=abs_wav, frame_length=1024, hop_length=256)[0]
    # E[log|z|]
    v2 = np.log(abs_wav).mean()
    v2_t = -rms(y=np.log(abs_wav), frame_length=1024, hop_length=256)[0]
    # log(E[|z|]) - E[log(|z|)]
    v3 = np.log(v1) - v2
    v3_t = np.log(v1_t) - v2_t

    wav_snr_idx = None
    if any(g_vals < v3):
        wav_snr_idx = np.where(g_vals < v3)[0].max()
    v3_t[0] = v3
    v3_t[-1] = v3
    t_idx = np.tile(g_vals, (len(v3_t),1)).T < v3_t
    t_idx = t_idx.sum(axis=0)-1
    t_idx[t_idx<0] = 0
    t_idx[t_idx>=len(db_vals)] = -1
    wav_snr_t = db_vals[t_idx]
    wav_snr_t[wav_snr_t==100] += v3_t[wav_snr_t==100] * 10
    wav_snr_t[wav_snr_t==-20] -= v3_t[wav_snr_t==-20] * 20
    
    if wav_snr_idx is None:
        wav_snr = db_vals[0]
        wav_snr -= v3 * 20
    elif wav_snr_idx == len(db_vals) - 1:
        wav_snr = db_vals[-1]
        wav_snr += v3 * 10
    else:
        wav_snr = db_vals[wav_snr_idx]

    return wav_snr, wav_snr_t