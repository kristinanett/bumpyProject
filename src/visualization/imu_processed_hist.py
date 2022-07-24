import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pandas_profiling import ProfileReport

csv_file_path = "data/processed/lowpass8/data.csv"
df = pd.read_csv(csv_file_path, header=0)
df_imu = df.iloc[:, 17:]
imu_all = np.array([df_imu]).flatten() #all 8 imu values for every image

#normalizing values
# imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
# imu_normalized = (imu_all-imu_mean)/imu_std
# imu_normalized = imu_normalized.flatten()

print(df_imu.describe())
print(df_imu[df_imu > 50].count()) #428, 424, 426, 427, 429, 428, 427, 431 (less than 1% of the whole data)

#pandas profiling (saved to bumpyproject/output.html and bumpyproject/output2.htmp)
# profile = ProfileReport(df, minimal=False)
# profile.to_file("output2.html")

#plotting with matplotlib
_ = plt.hist(imu_all, bins='auto', color = "#87CEBF")
plt.xlabel("Processed IMU values", fontsize=20)
plt.ylabel("Count", fontsize=20)
plt.grid()
plt.show()