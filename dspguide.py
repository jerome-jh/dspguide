#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

j = complex(0, 1)

def subplot(x, subplot, connected=False):
    ax = plt.subplot(subplot)
    if not connected:
        plt.plot(x, linestyle='', marker='s')
    else:
        plt.plot(x, linestyle='-', marker=None)
    ax.set_xlim(0, x.shape[0]-1)
    ax.grid()
    return ax

def show(x, connected=False):
    subplot(x, 111, connected)
    plt.show()

def ch2_fig21():
    N = 512
    s1 = np.random.normal(0.5, 1, N)
    s2 = np.random.normal(3, 0.2, N)
    ax = subplot(s1, 121, True)
    ax.set_ylim(-4, 8)
    ax = subplot(s2, 122, True)
    ax.set_ylim(-4, 8)
    plt.show()
    
def ch2_fig21b():
    show(s, True)
    
def ch2():
    ch2_fig21()

#ch2()
#quit()

def ch6_tab61_convolution(h, x):
    """
    Implementation of convolutation
    Input side algorithm as the author calls it
    M*N multiplications and additions
    with M and N number of items in h and x
    
    Returns: a N + M - 1 long signal
    """
    N = x.shape[0]
    M = h.shape[0]
    R = N + M -1
    y = np.zeros(R)
    for i in np.arange(N):
        for j in np.arange(M):
            k = i + j
            y[k] = y[k] + h[j] * x[i]
    return y

def ch6_eq61_convolution(h, x):
    """
    Implementation of convolution
    Using yet another technique to avoid overrun of x array: no padding, no test in inner loop
    Same number of multiplications and additions as other method, a bit more complexity
    
    Returns: a N + M - 1 long signal
    """
    N = x.shape[0]
    M = h.shape[0]
    ## Should swap arrays if not true
    assert(M <= N)
    R = N + M - 1
    y = np.zeros(R)
    nmac = 0
    for i in np.arange(R):
        lowj = np.amax([i - N + 1, 0])
        highj = np.amin([i + 1, M])
        for j in np.arange(lowj, highj):
            y[i] = y[i] + h[j] * x[i-j]
            nmac += 1
    print('M=%d N=%d #mac=%d'%(M, N, nmac))
    return y

def ch6_tab62_convolution_onlyvalid(h, x):
    """
    Implementation of convolution
    Returns only valid samples, where h is fully immersed in x
    (N - M + 1) * M  multiplications and additions
    with M and N number of items in h and x
    
    Returns: a N - M + 1 long signal
    """
    N = x.shape[0]
    M = h.shape[0]
    ## Should swap arrays if not true
    assert(M <= N)
    R = N - M + 1
    y = np.zeros(R)
    nmac = 0
    for i in np.arange(R):
        for j in np.arange(M):
            y[i] = y[i] + h[j] * x[i+(M-1)-j]
            nmac += 1
    print('M=%d N=%d #mac=%d'%(M, N, nmac))
    return y

def ch6_sample_signal():
    i = np.arange(60)
    s = np.zeros(81)
    s[10:70] = -np.sin(2*np.pi*np.arange(60)/20)
    s[10:70] += (0.5/10) * np.arange(60)
    return s

def ch6_ir_lowpass(l):
    i = np.arange(l+1)
    s = - i * (i - l)
    ## Normalize to one
    s = s / np.sum(s)
    return s

def ch6_ir_highpass(l):
    s = -ch6_ir_lowpass(l)
    s[l / 2] = 0
    s[l / 2] = -np.sum(s)
    return s

def ch6_ir_der(l=0):
    """
    Impulse response of a derivator
    """
    ir = np.zeros(l+2)
    ir[-1 + l/2] = 1
    ir[l/2] = -1
    return ir

def ch6_ir_invatt(l=0):
    """
    Impulse response of the inverting attenuator
    """
    ir = np.zeros(l+1)
    ir[l/2] = -0.5
    return ir

def ch6_fig63a_lowpass():
    ## Low pass filter
    h = ch6_ir_lowpass(30)
    x = ch6_sample_signal()
    y1 = ch6_tab61_convolution(h, x)
    y2 = ch6_eq61_convolution(h, x)
    y = np.convolve(h, x, mode='full')
    assert(np.allclose(y, y1))
    assert(np.allclose(y, y2))
    ax = subplot(x, 131)
    ax = subplot(h, 132)
    ax = subplot(y, 133)
    plt.show()

def ch6_fig63b_highpass():
    ## High pass filter
    h = ch6_ir_highpass(30)
    x = ch6_sample_signal()
    y1 = ch6_tab61_convolution(h, x)
    y2 = ch6_eq61_convolution(h, x)
    y = np.convolve(h, x, mode='full')
    assert(np.allclose(y, y1))
    assert(np.allclose(y, y2))
    ax = subplot(x,  131)
    ax = subplot(h,  132)
    ax = subplot(y, 133)
    plt.show()

def ch6_fig64b_inverting_attenuator():
    h = ch6_ir_invatt(28)
    x = ch6_sample_signal()
    y = np.convolve(h, x, mode='full')
    ax = subplot(x, 131)
    ax = subplot(h, 132)
    ax = subplot(y, 133)
    plt.show()

def ch6_fig64b_derivative():
    h = ch6_ir_der(28)
    x = ch6_sample_signal()
    y = np.convolve(h, x, mode='full')
    ax = subplot(x, 131)
    ax = subplot(h, 132)
    ax = subplot(y, 133)
    plt.show()

def ch6_fig610_border_effects():
    n = np.arange(0, 81)
    x = 2 + np.sin(2 * np.pi * n / 15)
    h = ch6_ir_highpass(30)
    y = np.convolve(h, x, mode='full')
    ax = subplot(x, 131)
    ax.set_ylim(-4, 4)
    subplot(h, 132)
    ax = subplot(y, 133)
    ax.set_ylim(-4, 4)
    plt.show()

def ch6_border_effects():
    n = np.arange(0, 81)
    x = 2 + np.sin(2 * np.pi * n / 15)
    h = ch6_ir_highpass(30)
    ## Output has same size as input
    y = np.convolve(h, x, mode='same')
    ax = subplot(x, 131)
    ax.set_ylim(-4, 4)
    subplot(h, 132)
    ax = subplot(y, 133)
    ax.set_ylim(-4, 4)
    plt.show()
    ## Output only valid samples
    y = np.convolve(h, x, mode='valid')
    y1 = ch6_tab62_convolution_onlyvalid(h, x)
    assert(np.allclose(y, y1))
    ax = subplot(x, 131)
    ax.set_ylim(-4, 4)
    subplot(h, 132)
    ax = subplot(y, 133)
    ax.set_ylim(-4, 4)
    plt.show()

def ch6():
    ch6_fig63a_lowpass()
    ch6_fig63b_highpass()
    ch6_fig64b_inverting_attenuator()
    ch6_fig64b_derivative()
    ## Example in the book
    ch6_fig610_border_effects()
    ## Personal tests
    ch6_border_effects()

ch6()
quit()

## Magnitude and phase of the transfer function
## Frequency between 0 and 1
## Magnitude is constant 1
## Phase varies linearily between 0 and 2*pi
def H_chirp(w):
    ph = 10 * w / 0.5
    return np.exp(j * ph)

def H_chirp2(w):
    ph = w / 0.5
    return np.exp(j * ph)

def H_integrator(w):
    H = np.zeros(w.shape, dtype=complex)
    for i in np.arange(w.shape[0]):
        if w[i] != 0:
            H[i] = 1 / (j*w[i])
        else:
            H[i] = 1e10
    return H

#tf_magnitude = np.ones((513))
#tf_phase = np.zeros((513))

#alpha = - 2 * np.pi / 512
#beta = 0
#
#for i in np.arange(tf_phase.shape[0]):
#    tf_phase[i] = alpha * i + beta * i * i

def plot_tf(H):
    """
    Plot magnitude and phase of transfer function
    Frequency between 0 and 1
    """
    f = np.linspace(-0.5, 0.5, 100)
    w = 2 * np.pi * f
    ax = plt.subplot(211)
    ax.plot(f, np.absolute(H(w)))
    ax = plt.subplot(212)
    ax.plot(f, np.angle(H(w)))
    plt.show()

def plot_ir(ir):
    """
    Plot an impulse response
    """
    plt.plot(ir, linestyle='', marker='s')
    plt.grid()
    plt.show()

def output_tf(H, x):
    """
    Compute the output of system with tranfer function H and input signal x
    using FFT convolution
    """
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    print(f)
    w = 2 * np.pi * f
    print(w)
    Y = np.multiply(H(w), X)
    y = np.fft.irfft(Y)
    return y

## Impulse response of an integrator
## Actually this is more a weighted average
def ir_int():
    return np.ones(10)

def output_ir(h, x):
    """
    Compute the output of system with impulse response h and input signal x
    using convolution
    """
    return np.convolve(h, x, mode='full')

def plot_input_output(x, y):
    ax = plt.subplot(211)
    ax.plot(x, linestyle='', marker='s')
    ax = plt.subplot(212)
    ax.plot(y, linestyle='', marker='s')
    plt.show()

## Make a Dirac
signal = np.zeros(100)
signal[10] = 1

#signal = np.arange(20)
#signal = np.sin(2*np.pi*0.1*signal)

## Compute the impulse response
#ir = np.convolve(tf, dirac, mode='valid')
#resp = np.convolve(tf, signal, mode='valid')
H = H_chirp
#H = H_integrator
plot_tf(H)
out = output_tf(H, signal)
#signal = sample_signal()
#out = output_tf(H, signal)
plot_input_output(signal, out)
quit()

out = output_ir(ir_der(), signal)
plot_input_output(signal, out)

out = output_ir(ir_int(), signal)
plot_input_output(signal, out)

