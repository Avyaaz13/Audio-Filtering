import math
def c_N(x, N):
    return math.cosh(N * math.acosh(x))
# Given parameters
delta = 0.15
delta1 = delta
delta2 = delta
Fs = 48 

# Sampling frequency in Hz
for i in range(11013, 11014):
    j = (i - 11000) % sum(int(digit) for digit in str(i))
    # print(f"i = {i}, j = {j}")
# Constants for passband and stopband frequencies
    Fp1 =  4 + 0.6*(j + 2)  # kHz
    Fp2 =  4 + 0.6*j  # kHz
    Fs1 = Fp1 + 0.3  # kHz
    Fs2 = Fp2 - 0.3  # kHz
    
    # Normalized digital filter frequencies
    wp1 = 2 * math.pi * Fp1 / Fs
    wp2 = 2 * math.pi * Fp2 / Fs
    ws1 = 2 * math.pi * Fs1 / Fs
    ws2 = 2 * math.pi * Fs2 / Fs
    
    # Center frequency
    wc = (wp1 + wp2) / 2
    
    # Analog filter parameters
    Omega_p1 = math.tan(wp1 / 2)
    Omega_p2 = math.tan(wp2 / 2)
    Omega_s1 = math.tan(ws1 / 2)
    Omega_s2 = math.tan(ws2 / 2)
    
    # Low pass filter parameters
    Omega_0 = math.sqrt(Omega_p1 * Omega_p2)
    B = Omega_p1 - Omega_p2
    OmegaLs1 = (Omega_s1**2 - Omega_0**2) / (B * Omega_s1)
    OmegaLs2 = (Omega_s2**2 - Omega_0**2) / (B * Omega_s2)
    Omega_Ls = min(abs(OmegaLs1), abs(OmegaLs2))
    
    # Function to calculate N and epsilon
    def calculate_N_epsilon(omega_Ls, delta1, delta2):
        D1 = 1 / ((1 - delta1)**2) - 1
        D2 = 1 / (delta2**2) - 1
        epsilonmax = math.sqrt(D1)
        N = math.ceil(math.acosh(math.sqrt(D2 / D1)) / math.acosh(omega_Ls))
        epsilonmin = math.sqrt(D2) / c_N(Omega_Ls, N)
        return N, epsilonmax, epsilonmin
    
    # Calculate N and epsilon
    N, epsilonmax, epsilonmin = calculate_N_epsilon(Omega_Ls, delta1, delta2)
    print(f"Roll No. = {i},j = {j},N = {N}, epsilonmax = {epsilonmax}, epsilonmin = {epsilonmin}")
