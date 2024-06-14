import os
import torch

# ディレクトリ構造を読み取り、フォルダ名とファイル名をマッピングする
def get_directory_structure(base_path):
    folder_names = []
    file_names = {}
    
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            folder_names.append(folder_name)
            file_names[folder_name] = os.listdir(folder_path)
    
    return folder_names, file_names

# 画像ファイルのパスを取得する関数
def get_image_paths(datasampler, folder_names, file_names, base_path):
    image_paths = []
    for idx in datasampler:
        idx = int(idx)  # idxをintに変換
        folder_index = idx // 100  # フォルダのインデックス
        image_index = idx % 100  # フォルダ内の画像のインデックス
        # print('folder_index:', folder_index)
        # print('image_index:', image_index)
        
        if folder_index < len(folder_names):
            folder_name = folder_names[folder_index]
            if image_index < len(file_names[folder_name]):
                image_name = file_names[folder_name][image_index]
                image_path = os.path.join(base_path, folder_name, image_name)
                image_paths.append(image_path)
                # print(image_path)
            else:
                print(f"Warning: Image index {image_index} out of range for folder {folder_name}")
        else:
            print(f"Warning: Folder index {folder_index} out of range")
    
    return image_paths