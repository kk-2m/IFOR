import torch
# WCSデータセット
color_train = torch.tensor([500, 500, 500, 500, 500, 500, 495, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 317, 500, 500, 500, 500, 500, 500])
infrared_train = torch.tensor([500, 500, 500, 337, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 350, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500])
# CCTデータセット, WCSデータセット
color_test = torch.tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
infrared_test = torch.tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])

# wcsのみ
torch.save(color_train, './dataset/wcs/dataset/color_train_sample_list.pt')
torch.save(infrared_train, './dataset/wcs/dataset/infrared_train_sample_list.pt')

# cct: ./dataset/cct/dataset/color_test_sample_list.pt
# wcs: ./dataset/wcs/dataset/color_test_sample_list.pt
torch.save(color_test, './dataset/wcs/dataset/color_test_sample_list.pt')
torch.save(infrared_test, './dataset/wcs/dataset/infrared_test_sample_list.pt')