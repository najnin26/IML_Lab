import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Set random seed for reproducibility
np.random.seed(42)


# Parameters
n_samples = 1000
time = np.arange(n_samples)
frequency = 0.05
amplitude = 10


# Generate synthetic time series (sine wave + noise)
signal = amplitude * np.sin(2 * np.pi * frequency * time)
noise = np.random.normal(0, 1.5, n_samples)
series = signal + noise


# Create pandas DataFrame
df = pd.DataFrame({'timestamp': pd.date_range(start='2020-01-01', periods=n_samples, freq='H'),
                  'value': series})


# Plot
plt.figure(figsize=(12, 4))
plt.plot(df['timestamp'], df['value'], label='Time Series')
plt.title('Synthetic Time Series Dataset')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
