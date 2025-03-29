import os

lq_dir = r"C:\Users\marka\Desktop\Deep-Fourier-Upsampling\Rain100H\rainy" 
for filename in os.listdir(lq_dir):
    if filename.endswith("_LQ.png"):  # 检查多余的 "_LQ" 后缀
        new_name = filename.replace("_LQ", "")  # 只保留一个 "_LQ"
        os.rename(os.path.join(lq_dir, filename), os.path.join(lq_dir, new_name))

print("LQ 文件名已修正！")
