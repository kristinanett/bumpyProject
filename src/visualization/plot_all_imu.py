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

csv_file_path = "data/processed/0405and1605and0106and1506new/data.csv" #'data/processed/data2.csv'
df = pd.read_csv(csv_file_path, header=0)
df_imu = df.iloc[:, 17:]
imu_all = np.array([df_imu]) #all 8 imu values for every image
imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
imu_normalized = (imu_all-imu_mean)/imu_std
imu_normalized = imu_normalized.flatten()

print(df_imu.describe())
print(df_imu[df_imu['imu1'] > 500].index)
print(df_imu[df_imu['imu2'] > 500].index)
print(df_imu[df_imu['imu3'] > 500].index)
print(df_imu[df_imu['imu4'] > 500].index)
print(df_imu[df_imu['imu5'] > 500].index)
print(df_imu[df_imu['imu6'] > 500].index)
print(df_imu[df_imu['imu7'] > 500].index)

_ = plt.hist(imu_all.flatten(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()