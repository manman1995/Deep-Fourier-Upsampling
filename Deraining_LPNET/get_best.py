log_path = './train_Rain13k-derain_nips_20211106_163626.log'
count = 0
line_num = 0
best_line_num = 0
best_psnr = 0
psnr_dict = {1:0,2:0,3:0}
best_psnr_dict = {1:0,2:0,3:0}
with open(log_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line_num += 1
        if line.find('Validation Rain13k,		 # psnr: ') > 0:
                index = line.find('Validation Rain13k,		 # psnr: ')
                off = len('Validation Rain13k,		 # psnr: ')
                psnr = line[index+off:]
                psnr = psnr[:-2]
                psnr = float(psnr)
                count += 1
                psnr_dict[count] = psnr
                if count % 3 == 0:
                    count = 0
                    if best_psnr < sum(psnr_dict.values())/3:
                        best_psnr = sum(psnr_dict.values())/3
                        best_psnr_dict[1] = psnr_dict[1]
                        best_psnr_dict[2] = psnr_dict[2]
                        best_psnr_dict[3] = psnr_dict[3]
                        best_line_num = line_num
print(best_psnr)
print(best_psnr_dict)
print(best_line_num)