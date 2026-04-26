import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("../results/results.csv", skipinitialspace=True)

# Distribution (Histogram)
plt.figure(figsize=(8,5))
df["mean_temp"].hist(bins=50)
plt.xlabel("Mean Temperature (°C)")
plt.ylabel("Number of Buildings")
plt.title("Distribution of Mean Building Temperatures")
plt.savefig("mean_temperature_hist.png", dpi=200)
plt.show()

# Average mean temperature
avg_mean_temp = df["mean_temp"].mean()
print("Average mean temperature:", avg_mean_temp)

# Average temperature std
avg_std = df["std_temp"].mean()
print("Average temperature std:", avg_std)

# Buildings with >=50% above 18°C
above18_count = (df["pct_above_18"] >= 50).sum()
print("Buildings >=50% area above 18°C:", above18_count)

# Buildings with >=50% below 15°C
below15_count = (df["pct_below_15"] >= 50).sum()
print("Buildings >=50% area below 15°C:", below15_count)