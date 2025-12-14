import pandas as pd

df_w = pd.read_csv("253/metrics_wiener.csv")
df_r = pd.read_csv("253/metrics_rl.csv")

mean_w = df_w[["psnr", "ssim", "niqe", "lpips"]].mean()
mean_r = df_r[["psnr", "ssim", "niqe", "lpips"]].mean()

df_w.loc[len(df_w.index)] = ["AVERAGE", "-", mean_w["psnr"], mean_w["ssim"], mean_w["niqe"], mean_w["lpips"]]
df_r.loc[len(df_r.index)] = ["AVERAGE", "-", mean_r["psnr"], mean_r["ssim"], mean_r["niqe"], mean_r["lpips"]]

df_w.to_csv("253/metrics_wiener_with_mean.csv", index=False)
df_r.to_csv("253/metrics_rl_with_mean.csv", index=False)

print("Mean rows added!\nSaved to:")
print("metrics_wiener_with_mean.csv")
print("metrics_rl_with_mean.csv")
