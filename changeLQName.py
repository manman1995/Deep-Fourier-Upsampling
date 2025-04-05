import os

# 设置目标文件夹路径
target_dir = r"/scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/Deep-Fourier-Upsampling/Dataset/RainTrainH_modified/RainTrainH_modified/norain"  # 修改为你的实际本地路径

# 遍历目录下的所有文件
for filename in os.listdir(target_dir):
    if filename.startswith("norain-") and filename.endswith(".png"):
        new_name = filename.replace("norain-", "rain-", 1)
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_name)
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_name}")

print("所有文件名已成功替换！")
