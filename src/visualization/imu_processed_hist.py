import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pandas_profiling import ProfileReport

csv_file_path = "data/processed/lowpass8/data.csv"
df = pd.read_csv(csv_file_path, header=0)
df_imu1 = df.iloc[:12952, 17]
df_imu2 = df.iloc[12952:23074, 17]
df_imu3 = df.iloc[23074:29392, 17]
df_imu4 = df.iloc[29392:, 17]

imu_all1 = np.array([df_imu1]).flatten()
imu_all2 = np.array([df_imu3]).flatten()
imu_all3 = np.array([df_imu3]).flatten()
imu_all4 = np.array([df_imu4]).flatten()

#normalizing values
# imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
# imu_normalized = (imu_all-imu_mean)/imu_std
# imu_normalized = imu_normalized.flatten()

print(df.iloc[:, 17:].describe())
print(df.iloc[:, 17:][df.iloc[:, 17:] < -20].count()) #428, 424, 426, 427, 429, 428, 427, 431 (less than 1% of the whole data)

#pandas profiling (saved to bumpyproject/output.html and bumpyproject/output2.htmp)
# profile = ProfileReport(df, minimal=False)
# profile.to_file("output2.html")

df1 = pd.DataFrame(np.array([df_imu4]).flatten(), columns=["day4"])
df2 = pd.DataFrame(np.array([df_imu1]).flatten(), columns=["day1"])
df3 = pd.DataFrame(np.array([df_imu2]).flatten(), columns=["day2"])
df4 = pd.DataFrame(np.array([df_imu3]).flatten(), columns=["day3"])
df_plot = pd.concat([df1, df2, df3, df4], axis = 1)

# plot the data
ax = df_plot.plot.hist(stacked=False, bins=500, density=False, figsize=(10, 6), grid=False, color=['#60A7E2', '#87CEBF', '#E27396', '#D7D85F'])
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.set_axisbelow(True)
ax.set_ylabel("Frequency", fontsize = 12)
ax.set_xlabel("Processed IMU values", fontsize = 12)
ax.grid(axis='y', color='grey', ls=':')
plt.show()