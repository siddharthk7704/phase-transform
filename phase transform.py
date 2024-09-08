import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

# Define the message signal parameters
amplitude = 1  # Amplitude of the message signal
sampling_rate = 100  # Sampling rate in Hz
start_time = -5
end_time = 5
alpha = float(input("Enter the phase angle in degree by which you want to apply shift to the message signal: "))
# Create the time array
t = np.linspace(start_time, end_time, int((end_time - start_time) * sampling_rate), endpoint=False)

# Define the message signal (example: sinusoidal signal)
message_signal = np.cos(t)

# Compute the Fourier Transform of the message signal
message_fft = fft(message_signal)

# Frequency array
freqs = fftfreq(len(message_signal), 1/sampling_rate)

# Define the scaling factors
cos_scaling_factor = np.cos(np.deg2rad(alpha))
sin_scaling_factor = np.sin(np.deg2rad(alpha))

# Create the -j*sgn(f) term
j_sgn_f = -1j * np.sign(freqs)

# Multiply the frequency domain representations
result_fft = message_fft * (cos_scaling_factor + sin_scaling_factor * j_sgn_f)

# Compute the inverse Fourier Transform to obtain the result in the time domain
result_signal = ifft(result_fft)

# Plot both the original message signal and the resulting signal in the time domain using subplots
plt.figure(figsize=(12, 10))

# Plot the original message signal
plt.subplot(2, 1, 1)
plt.plot(t, message_signal, label='Original Signal', color='blue')
plt.title('Original Message Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(-5, 5)
plt.grid(True)
plt.legend()

# Plot the resulting signal
plt.subplot(2, 1, 2)
plt.plot(t, np.real(result_signal), label='Resulting Signal', color='red')
plt.title('Resulting Signal After Applying Phase shift')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(-5, 5)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
