import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# mu, sigma = 0, 1 # mean and standard deviation
# s = np.random.normal(mu, sigma, 100000)
# s2 = np.square(s)
# print(np.mean(s2), "/t", np.std(s2))


# _ = plt.hist(s, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()

# _ = plt.hist(s2, bins='auto')  # arguments are passed to np.histogram
# plt.title("Histogram with 'auto' bins")
# plt.show()


csv_file_path = 'data/processed/data2.csv'
df = pd.read_csv(csv_file_path, header=0)
imu_all = np.array([df.iloc[:, 16:]])
imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
imu_standard = (imu_all-imu_mean)/imu_std

imu_standard = imu_standard.flatten()

_ = plt.hist(imu_all.flatten(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()