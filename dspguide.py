#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

j = complex(0, 1)

## Magnitude and phase of the transfer function
## Frequency between 0 and 1
## Magnitude is constant 1
## Phase varies linearily between 0 and 2*pi
def H_chirp(w):
    ph = 250 * w / 0.5
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

## Impulse response of a derivator
def ir_der(l=0):
    ir = np.zeros(l+2)
    ir[-1 + l/2] = 1
    ir[l/2] = -1
    return ir

## Impulse response of an integrator
## Actually this is more a weighted average
def ir_int():
    return np.ones(10)

def ir_lowpass(l):
    i = np.arange(l+1)
    s = - i * (i - l)
    ## Normalize to one
    s = s / np.sum(s)
    return s

def ir_highpass(l):
    s = -ir_lowpass(l)
    s[l / 2] = 0
    s[l / 2] = -np.sum(s)
    return s


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
    w = 2 * np.pi * f
    Y = np.multiply(H(w), X)
    y = np.fft.irfft(Y)
    return y

def output_ir(h, x):
    """
    Compute the output of system with impulse response h and input signal x
    using convolution
    """
    return np.convolve(h, x, mode='full')

def plot_signal(x):
    plt.plot(x, linestyle='', marker='s')
    plt.show()

def plot_input_output(x, y):
    ax = plt.subplot(211)
    ax.plot(x, linestyle='', marker='s')
    ax = plt.subplot(212)
    ax.plot(y, linestyle='', marker='s')
    plt.show()

def sample_signal():
    i = np.arange(60)
    s = np.zeros(81)
    s[10:70] = -np.sin(2*np.pi*np.arange(60)/20)
    s[10:70] += (0.5/10) * np.arange(60)
    return s

def ex_6_3_lowpass():
    ## Low pass filter
    ir = ir_lowpass(30)
    plot_ir(ir)
    out = output_ir(ir, sample_signal())
    plot_input_output(sample_signal(), out)

def ex_6_3_highpass():
    ## High pass filter
    ir = ir_highpass(30)
    plot_ir(ir)
    out = output_ir(ir, sample_signal())
    plot_input_output(sample_signal(), out)

def ex_6_4_derivator():
    ir = ir_der(30)
    out = output_ir(ir, sample_signal())
    plot_input_output(sample_signal(), out)

#ex_6_4_derivator()
#quit()

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

