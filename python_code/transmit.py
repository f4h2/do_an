import numpy as np
from gnss_utils import generateCAcode

def transmit():
    speedOfLight = 299792458
    samplingFreq = 10e6
    codeLength = 1023
    codeRate = 1.023e6
    NumTr = 3
    numSamp = int(0.01 * samplingFreq)
    
    prnCode = np.zeros((3, 2046))
    for i in range(3):
        caCode = generateCAcode(i + 1)
        prnCode[i, :] = np.concatenate([caCode, caCode])
        
    Xt = np.array([[0, 1],
                   [549, 0],
                   [0, 545]])
    
    Xr = np.array([430, 280])
    
    codePhase = codeRate / samplingFreq
    
    R = np.zeros(NumTr)
    Rt = np.zeros(NumTr)
    Rn = np.zeros(NumTr)
    
    for i in range(NumTr):
        R[i] = np.sqrt((Xt[i, 0] - Xr[0])**2 + (Xt[i, 1] - Xr[1])**2)
        Rt[i] = R[i] / speedOfLight
        Rn[i] = Rt[i] * samplingFreq
        
    signal_1Block = np.zeros(numSamp)
    t_idx = np.arange(numSamp)
    for i in range(NumTr):
        tcode = t_idx * codePhase - Rt[i] * codeRate
        tcode2 = np.floor(np.mod(tcode + 1023, 1023)).astype(int)
        code = prnCode[i, tcode2]
        signal_1Block = signal_1Block + code
        
    signal = 1024 * signal_1Block
    signalIQ = np.zeros(2 * len(signal), dtype=np.int16)
    signalIQ[0::2] = signal.astype(np.int16)
    
    with open('test.bin', 'wb') as fid:
        fid.write(signalIQ.tobytes())

if __name__ == "__main__":
    transmit()
