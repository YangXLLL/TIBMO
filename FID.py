import torch
import torchvision

from pytorch_fid import fid_score

# 准备真实数据分布和生成模型的图像数据
real_images_folder = 'image_quality/images/image_clean'
# real_images_folder = 'images/image1k'
#generated_images_folder1 = 'experiment/output-TCL-shunxu-text30'
generated_images_folder1 = 'output-fuxian/1'
# generated_images_folder1 = 'output-TCL-shunxu-text666'
# generated_images_folder1 = 'output-ALBEF-shunxu-text-sga666'

# generated_images_folder1 = 'images/TCL/image_clean-TCL'
# generated_images_folder1 = 'output-fuxian/1'
# generated_images_folder1 = 'images/image_MDA'
# generated_images_folder1 = 'images/image_PGD'
# generated_images_folder1 = 'images/image_Co-attack'
# generated_images_folder1 = 'images/image_SGA'


# generated_images_folder2 = 'images/image_MDA'
# generated_images_folder3 = 'images/image_PGD'
# generated_images_folder4 = 'images/image_Co-attack'
# generated_images_folder = 'images/image_SGA'

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)



# 计算FID距离值
# fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
#                                                  inception_model,
#                                                  transform=transform)
fid_value1 = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder1],batch_size=50,device=torch.device("cuda"), dims=2048)
# fid_value2 = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder2],batch_size=50,device=torch.device("cuda"), dims=2048)
# fid_value3 = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder3],batch_size=50,device=torch.device("cuda"), dims=2048)
# fid_value4 = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder4],batch_size=50,device=torch.device("cuda"), dims=2048)
print('FID value1:', fid_value1)
# print('FID value2:', fid_value2)
# print('FID value3:', fid_value3)
# print('FID value4:', fid_value4)

