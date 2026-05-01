import numpy as np
from gnss_utils import generateCAcode

def receiver():
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
    
    with open('test.bin', 'rb') as fid:
        signalIQ = np.frombuffer(fid.read(2 * numSamp * 2), dtype=np.int16)
        
    signal = signalIQ[0::2].astype(float)
    
    # Receiver Processing
    numProcessedSamp = 50000
    codePhase = codeRate / samplingFreq
    
    Xcorr = np.zeros((NumTr, int(codeLength / codeRate * samplingFreq) + 1))
    Tn = np.zeros(NumTr)
    
    tau_max = int(codeLength / codeRate * samplingFreq)
    for i in range(NumTr):
        for tau in range(tau_max + 1):
            tcode = (np.arange(numProcessedSamp) + tau) * codePhase
            tcode2 = np.floor(np.mod(tcode, 1023)).astype(int)
            code = prnCode[i, tcode2]
            Xcorr[i, tau] = np.sum(code * signal[:numProcessedSamp])**2
            
        idx = np.argmax(Xcorr[i, :])
        Tn[i] = idx
        print(f"{idx} of sv {i+1}")
        
    T = np.zeros(NumTr)
    P = np.zeros(NumTr)
    
    for i in range(NumTr):
        T[i] = np.max(Tn) - Tn[i]
        P[i] = (5 + T[i]) / samplingFreq * speedOfLight
        
    pos = np.zeros(3)
    obs = P
    numOfIterations = 10
    F = np.zeros(NumTr)
    J = np.zeros((NumTr, 3))
    
    for iter in range(numOfIterations):
        for i in range(NumTr):
            F[i] = obs[i] - np.sqrt((Xt[i, 0] - pos[0])**2 + (Xt[i, 1] - pos[1])**2) - pos[2]
            J[i, 0] = (-(Xt[i, 0] - pos[0])) / obs[i]
            J[i, 1] = (-(Xt[i, 1] - pos[1])) / obs[i]
            J[i, 2] = 1
            
        # delta_pos = J \ F
        delta_pos, _, _, _ = np.linalg.lstsq(J, F, rcond=None)
        pos = pos + delta_pos
        
    print(f"Vị trí bộ thu: {pos[:2]}")

if __name__ == "__main__":
    receiver()
