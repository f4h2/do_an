import math
import numpy as np

def generateCAcode(prn):
    """
    Generates 1023-bit C/A code for a given PRN (1 to 32).
    Output maps 0 to 1 and 1 to -1.
    """
    g2_taps = [
        [2, 6], [3, 7], [4, 8], [5, 9], [1, 9],
        [2, 10], [1, 8], [2, 9], [3, 10], [2, 3],
        [3, 4], [5, 6], [6, 7], [7, 8], [8, 9],
        [9, 10], [1, 4], [2, 5], [3, 6], [4, 7],
        [5, 8], [6, 9], [1, 3], [4, 6], [5, 7],
        [6, 8], [7, 9], [8, 10], [1, 6], [2, 7],
        [3, 8], [4, 9]
    ]
    if prn < 1 or prn > 32:
        raise ValueError("PRN must be between 1 and 32")
    
    tap1, tap2 = g2_taps[prn - 1]
    tap1 -= 1
    tap2 -= 1
    
    g1 = [1] * 10
    g2 = [1] * 10
    
    ca_code = np.zeros(1023, dtype=int)
    
    for i in range(1023):
        # Calculate output bit
        g2_out = g2[tap1] ^ g2[tap2]
        out_bit = g1[9] ^ g2_out
        ca_code[i] = 1 if out_bit == 0 else -1
        
        # Calculate feedback
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        
        # Shift
        g1 = [fb1] + g1[:-1]
        g2 = [fb2] + g2[:-1]
        
    return ca_code

def calcDistance(lat1, lon1, lat2, lon2):
    """
    Calculates distance between two geographic coordinates using Haversine formula.
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    R = 6371000 # Radius of Earth in meters
    distance = R * c
    return distance
