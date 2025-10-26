import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 假设你已有这三个模型的定义
from models.low_level_vision import LowLevelVisionModel
from models.mid_level_temporal import MidLevelTemporalModel
from models.high_level_semantic import HighLevelSemanticModel
from fusion_classifier import FusionClassifier

# 1. 定义你的数据集
class FusionDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: List[dict], 每个dict包含
            {
                'video_path': ...,
                'label': 0/1
            }
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        return item['video_path'], item['label']

# 2. 提取特征的函数
# 你需要根据你的模型接口调整此函数
# 假设 process_video 返回一维特征向量
@torch.no_grad()
def extract_features(video_path, low_model, mid_model, high_model, device):
    frames, reason = low_model.preprocessor.extract_keyframes(video_path, num_frames=16)
    if frames is None or not isinstance(frames, torch.Tensor) or frames.numel() == 0:
        return None, None, None
    frames = frames.unsqueeze(0).to(device)  # [1, 16, 3, 224, 224]
    # 低层特征
    raw_feats = low_model.features(frames[0])  # [16, 2048, 1, 1]
    low_feats = raw_feats.squeeze(-1).squeeze(-1)  # [16, 2048]
    low_feat = low_feats.mean(dim=0)  # [2048]
    # 中层特征
    mid_input = low_feats.unsqueeze(0).contiguous()  # [1, 16, 2048]
    mid_out = mid_model(mid_input).squeeze(0)  # [16, 2048]
    mid_feat = mid_out.mean(dim=0)  # [2048]
    # 高层特征
    high_out = high_model(mid_input)
    if high_out.dim() == 3:
        high_out = high_out.squeeze(0)
    if high_out.dim() == 2:
        high_feat = high_out.mean(dim=0)
    elif high_out.dim() == 1:
        high_feat = high_out.mean().item()  # 标量 float
    else:
        high_feat = high_out.mean().item()
    return low_feat, mid_feat, high_feat

# 3. 训练主流程
def train_fusion_layer(
    train_data, val_data=None, 
    low_weight_path='best_models_pth/best_low_level_vision.pth',
    mid_weight_path='best_models_pth/mid_level_temporal_best.pth',
    high_weight_path='best_models_pth/high_level_semantic_best.pth',
    fusion_save_path='best_models_pth/fusion_classifier_best.pth',
    epochs=10, batch_size=8, lr=1e-3, device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # 加载三个子模型
    low_model = LowLevelVisionModel().to(device)
    low_model.load_state_dict(torch.load(low_weight_path, map_location=device))
    mid_model = MidLevelTemporalModel(input_dim=2048).to(device)
    mid_model.load_state_dict(torch.load(mid_weight_path, map_location=device))
    high_model = HighLevelSemanticModel().to(device)
    high_model.load_state_dict(torch.load(high_weight_path, map_location=device))
    # 冻结参数
    for m in [low_model, mid_model, high_model]:
        for p in m.parameters():
            p.requires_grad = False

    # 初始化融合模型
    fusion_model = FusionClassifier().to(device)
    # 替换子模型
    fusion_model.low_level_model = low_model
    fusion_model.mid_level_model = mid_model
    fusion_model.high_level_model = high_model

    # 只训练融合层参数
    optimizer = optim.Adam([
        {'params': fusion_model.first_layer_weights},
        {'params': fusion_model.second_layer.parameters()}
    ], lr=lr)
    criterion = nn.BCELoss()

    # 数据集
    train_dataset = FusionDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    fusion_model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_valid = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (video_paths, labels) in enumerate(pbar):
            batch_low, batch_mid, batch_high = [], [], []
            valid_indices = []
            for idx, vp in enumerate(video_paths):
                low, mid, high = extract_features(vp, low_model, mid_model, high_model, device)
                if low is not None and mid is not None and high is not None:
                    if hasattr(high, 'dim') and (high.dim() > 0 and high.shape[0] > 1):
                        high = high.mean().item()
                    elif torch.is_tensor(high):
                        high = high.item() if high.numel() == 1 else float(high)
                    batch_low.append(low)
                    batch_mid.append(mid)
                    batch_high.append(high)
                    valid_indices.append(idx)
            if not batch_low:
                continue
            batch_low = torch.stack(batch_low).to(device)
            batch_mid = torch.stack(batch_mid).to(device)
            batch_high = torch.tensor(batch_high, dtype=torch.float32).to(device)  # [batch]
            labels = labels[valid_indices].float().to(device)
            total_valid += len(labels)

            # 前向
            first_layer_weights = torch.softmax(fusion_model.first_layer_weights, dim=0)
            first_layer_output = torch.cat([
                batch_low.mean(dim=1, keepdim=True),
                batch_mid.mean(dim=1, keepdim=True),
                batch_high.unsqueeze(1)
            ], dim=1)  # [batch, 3]
            first_layer_output = first_layer_weights * first_layer_output
            outputs = fusion_model.second_layer(first_layer_output).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader.dataset):.4f}")

    # 保存融合层权重
    torch.save(fusion_model.state_dict(), fusion_save_path)
    print(f"融合层权重已保存到: {fusion_save_path}")

# 4. 入口
if __name__ == "__main__":
    # 你需要准备训练数据列表
    # 例如 [{'video_path': 'data/xxx.mp4', 'label': 1}, ...]
    import json
    with open('train_fusion_data.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    train_fusion_layer(train_data, 
        low_weight_path='best_models_pth/best_low_level_vision.pth',
        mid_weight_path='best_models_pth/mid_level_temporal_best.pth',
        high_weight_path='best_models_pth/high_level_semantic_best.pth',
        fusion_save_path='best_models_pth/fusion_classifier_best.pth',
        epochs=10, batch_size=8, lr=1e-3) 