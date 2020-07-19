import numpy as np
from time import perf_counter

## FT of (2/w)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*((v-v0)/w)**2)*dv
gG_FT = lambda x, w: np.exp(-(np.pi*x*w)**2/(4*np.log(2)))


## FT of (1/np.pi) * (w/2) / ((v-v0)**2 + (w/2)**2)*dv
gL_FT = lambda x, w: np.exp(-np.pi*x*w)


def init_axes(v,NwG_in,NwL_in,log_wGi,log_wLi,eps = 1e-4):
    global Nv,dv,v_min
    Nv = v.size
    dv = (v[-1] - v[0]) / (Nv - 1)
    v_min = np.min(v)

    global NwG,log_dwG,log_wG_min,swG
    NwG = NwG_in
    log_wG_min = np.min(log_wGi)
    log_wG_max = np.max(log_wGi) + eps
    log_dwG = (log_wG_max - log_wG_min) / (NwG - 1)
    swG = 1 / (1 - np.exp(-log_dwG))
	
    global NwL,log_dwL,log_wL_min,swL
    NwL = NwL_in
    log_wL_min = np.min(log_wLi)
    log_wL_max = np.max(log_wLi) + eps
    log_dwL = (log_wL_max - log_wL_min) / (NwL - 1)
    swL = 1 / (1 - np.exp(-log_dwL))

    
## Calculate Spectral matrix:
def calc_spectral_matrix(v0i, log_wGi, log_wLi, S0i):

    #  Initialize matrix:
    S_klm = np.zeros((2 * Nv, NwG, NwL),dtype=np.float32)

    ki  = (v0i - v_min)/dv
    ki0 = ki.astype(int)
    ki1 = ki0 + 1
    tvi = ki - ki0 

    li  = (log_wGi - log_wG_min)/log_dwG
    li0 = li.astype(int)
    li1 = li0 + 1
    twGi = li - li0 

    mi  = (log_wLi - log_wL_min)/log_dwL
    mi0 = mi.astype(int)
    mi1 = mi0 + 1
    twLi = mi - mi0 

    # Calculate weights:
    avi  =                   tvi
    awGi = swG * (1 - np.exp(-log_dwG * twGi))
    awLi = swL * (1 - np.exp(-log_dwL * twLi))

    awV00 = (1-awGi) * (1-awLi) 
    awV01 = (1-awGi) *    awLi  
    awV10 =    awGi  * (1-awLi) 
    awV11 =    awGi  *    awLi
    
    Sv0   = (1-avi)  *     S0i
    Sv1   =    avi   *     S0i

    # Add lines to spectral matrix:
    np.add.at(S_klm, (ki0, li0, mi0), Sv0 * awV00)
    np.add.at(S_klm, (ki0, li0, mi1), Sv0 * awV01)
    np.add.at(S_klm, (ki0, li1, mi0), Sv0 * awV10)
    np.add.at(S_klm, (ki0, li1, mi1), Sv0 * awV11)
    np.add.at(S_klm, (ki1, li0, mi0), Sv1 * awV00)
    np.add.at(S_klm, (ki1, li0, mi1), Sv1 * awV01)
    np.add.at(S_klm, (ki1, li1, mi0), Sv1 * awV10)
    np.add.at(S_klm, (ki1, li1, mi1), Sv1 * awV11)
    
    return S_klm


## Apply transform:
def forward_transform(S_klm):
    
    S_klm_FT = np.zeros((NwG,NwL,Nv+1), dtype = np.complex64)
    for l in range(NwG):
        for m in range(NwL):
            S_klm_FT[l,m] = np.fft.rfft(S_klm[:,l,m])
            
    return S_klm_FT


def multiply_lineshapes(S_klm_FT):

    wG = np.exp(log_wG_min + log_dwG * np.arange(NwG))
    wL = np.exp(log_wL_min + log_dwL * np.arange(NwL))
    
    x     = np.arange(Nv + 1) / (2 * Nv * dv)
    IG_FT = [gG_FT(x,wG[l]) for l in np.arange(NwG)]
    IL_FT = [gL_FT(x,wL[m]) for m in np.arange(NwL)]
    
    S_k_FT = np.zeros(Nv + 1, dtype = np.complex64)  
    for l in range(NwG):
        for m in range(NwL):
            S_k_FT += S_klm_FT[l,m] * IG_FT[l] * IL_FT[m]

    return S_k_FT


def reverse_transform(S_k_FT):
    return np.fft.irfft(S_k_FT)[:Nv]/dv


## Synthesize spectrum:
def spectrum(v, NwG, NwL, v0i, log_wGi, log_wLi, S0i):

    # Initialize width-axes:
    t1 = perf_counter()
    init_axes(v,NwG,NwL,log_wGi,log_wLi)
##    print('1',perf_counter()-t1)

    # Calculate spectral matrix:
    t2 = perf_counter()
    S_klm = calc_spectral_matrix(v0i, log_wGi, log_wLi, S0i)
##    print('2',perf_counter() - t2)

    # Forward transform:
    t3 = perf_counter()
    S_klm_FT = forward_transform(S_klm)
##    print('3',perf_counter() - t3)

    # Apply lineshapes:
    t4 = perf_counter()    
    S_k_FT = multiply_lineshapes(S_klm_FT)
##    print('4',perf_counter()-t4)

    # Reverse transform:
    t5 = perf_counter()
    I = reverse_transform(S_k_FT)
##    print('5',perf_counter()-t5)
       
    return I,S_klm
