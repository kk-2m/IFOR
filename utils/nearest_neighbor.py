import torch
import numpy as np

def k_center(features, groups:int, device="cuda"):
    c = torch.zeros([groups, features.shape[1]], device=device)
    id_size = torch.zeros(groups, device=device)
    distance = torch.zeros([features.shape[0],groups], device=device)
    min_dist = torch.zeros(1, device=device)+9999
    best_id = torch.zeros(features.shape[0], device=device)
    best_c = c
    for big_epoch in range(20):
        # 各要素をランダムにクラスタに割り当てる（k=50）
        f_id = torch.randint(groups, size=[features.shape[0]],dtype=torch.int32, device=device)
        for epoch in range(30):
            # 要素の平均をそのクラスタ中心とする
            for k in range(groups):
                id_size[k] = torch.sum(f_id==k)
                if id_size[k] != 0:
                    c[k,:] = torch.mean(features[f_id==k,:],dim=0)
            # 各要素とクラスタ中心の距離を計算し、最も近いクラスタに再割り当て
            for k in range(groups):
                distance[:,k] = torch.sum(torch.pow((features - c[k,:]),2),dim = 1)
                new_id = torch.argsort(distance,dim = 1)[:,0].type(torch.int32)
            # 割り当てに変化がなくなったらループを終了
            if torch.sum(torch.abs(f_id-new_id))==0:
                break
            else:
                f_id=new_id
        total_dist = torch.zeros(1, device=device)
        for k in range(groups):
            total_dist += torch.mean(torch.sqrt(distance[f_id==k,k]))
        # 値の更新
        if total_dist<min_dist:
            min_dist = total_dist
            best_id = f_id
            best_c = c
    # 最良のクラスタ割り当てに基づいて、各クラスタのサイズを計算
    for k in range(groups):
        id_size[k] = torch.sum(best_id==k)
    print('best_c', best_c)
    return best_id, best_c

# クラスタ中心を更新する関数
def k_center_simple(features, f_id, c, groups:int, device="cpu"):
	id_size = torch.zeros(groups, device=device)
	distance = torch.zeros([features.shape[0],groups], device=device)

	for epoch in range(40):
		for k in range(groups):
			id_size[k] = torch.sum(f_id==k)
			if id_size[k] != 0:
                # 各特徴次元方向に対して平均を求める
				c[k,:] = torch.mean(features[f_id==k,:],dim=0)
		for k in range(groups):
			distance[:,k] = torch.sum(torch.pow((features - c[k,:]),2),dim = 1)
			new_id = torch.argsort(distance,dim = 1)[:,0].type(torch.int32)
		if torch.sum(torch.abs(f_id-new_id))==0:
			break
		else:
			f_id=new_id
			
	for k in range(groups):
		id_size[k] = torch.sum(f_id==k)

	print('c', c)
	print('id_size=',id_size.type(torch.int32))
	print(np.where(id_size.cpu()!=0)[0].shape)
	
	return f_id, c, id_size