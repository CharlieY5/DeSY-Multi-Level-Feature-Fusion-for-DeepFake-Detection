import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os

class VideoPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_keyframes(self, video_path, num_frames=16):
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return torch.empty(0), '视频文件不存在，可能上传失败或路径错误'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"视频无法打开: {video_path}")
            return torch.empty(0), '视频无法打开，文件可能损坏或格式不支持'
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            print(f"视频帧数为0: {video_path}")
            return torch.empty(0), '视频帧数为0，文件可能损坏或格式不支持'
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frame = self.transform(frame)
                frames.append(frame)
        cap.release()
        if not frames:
            print(f"未能提取到任何帧: {video_path}")
            return torch.empty(0), '未能提取到任何帧，视频可能损坏或格式不支持'
        return torch.stack(frames), None

class ResNet50FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet50FeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.conv2 = resnet.layer1
        self.conv3 = resnet.layer2
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames = x.size(0), x.size(1)
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        
        # Reshape back to include temporal dimension
        f1 = f1.view(batch_size, num_frames, -1)
        f2 = f2.view(batch_size, num_frames, -1)
        f3 = f3.view(batch_size, num_frames, -1)
        
        return f1, f2, f3

class LowLevelVisionModel(nn.Module):
    def __init__(self, weight_path=None):
        super(LowLevelVisionModel, self).__init__()
        if weight_path is not None:
            self.resnet = models.resnet50(weights=None)
            state_dict = torch.load(weight_path, map_location='cpu')
            # 兼容多种权重结构
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            # 去除常见前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                key = k
                for prefix in ['resnet.', 'features.', 'module.', 'model.']:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                if key.startswith(('conv1', 'bn1', 'layer', 'downsample', 'fc')):
                    new_state_dict[key] = v
            missing, unexpected = self.resnet.load_state_dict(new_state_dict, strict=False)
            print(f"[LowLevelVisionModel] 权重加载完成，missing={missing}, unexpected={unexpected}")
        else:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 添加新的分类层
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 修改为2个类别（真实/生成）
        )
        
        # 初始化视频预处理器
        self.preprocessor = VideoPreprocessor()
        
    def forward(self, x):
        # 输入x的形状应该是 [batch_size, num_frames, channels, height, width]
        batch_size, num_frames, c, h, w = x.size()
        
        # 将输入重塑为 [batch_size * num_frames, channels, height, width]
        x = x.view(-1, c, h, w)
        
        # 提取特征
        features = self.features(x)
        features = features.view(batch_size, num_frames, -1)
        
        # 对每一帧的特征取平均
        features = features.mean(dim=1)
        
        # 分类
        output = self.classifier(features)
        return output  # 返回形状为 [batch_size, 2] 的张量
    
    def process_video(self, video_path, num_frames=16):
        """处理视频文件并返回检测结果"""
        try:
            # 提取关键帧
            frames, reason = self.preprocessor.extract_keyframes(video_path, num_frames)
            print(f"[LOWLEVEL DEBUG] video_path={video_path}, frames_type={type(frames)}, frames_shape={getattr(frames, 'shape', None)}, reason={reason}")
            if frames is None or not isinstance(frames, torch.Tensor) or frames.numel() == 0:
                print(f"视频帧提取失败: {video_path}")
                return {
                    'is_ai_generated': False,
                    'confidence': 0.0,
                    'feature_scores': {
                        'low_level': 0.0,
                        'mid_level': 0.0,
                        'high_level': 0.0
                    },
                    'reason': reason or '视频帧提取失败，可能为无效或损坏视频'
                }
            # 添加batch维度
            frames = frames.unsqueeze(0)  # [1, num_frames, channels, height, width]
            print(f"[LOWLEVEL DEBUG] frames after unsqueeze: {frames.shape}")
            # 设置为评估模式
            self.eval()
            with torch.no_grad():
                # 前向传播
                output = self.forward(frames)
                print(f"[LOWLEVEL DEBUG] forward输出: {output}")
                # 应用softmax获取概率
                probabilities = torch.softmax(output, dim=1)
                print(f"[LOWLEVEL DEBUG] softmax概率: {probabilities}")
                # 获取AI生成的概率
                ai_probability = probabilities[0, 1].item()
                return {
                    'is_ai_generated': ai_probability > 0.5,
                    'confidence': ai_probability,
                    'feature_scores': {
                        'low_level': ai_probability,
                        'mid_level': ai_probability,
                        'high_level': ai_probability
                    },
                    'reason': '推理成功'
                }
        except Exception as e:
            print(f"处理视频时出错: {e}")
            return {
                'is_ai_generated': False,
                'confidence': 0.0,
                'feature_scores': {
                    'low_level': 0.0,
                    'mid_level': 0.0,
                    'high_level': 0.0
                },
                'reason': f'模型推理异常: {e}'
            }

if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    num_frames = 16
    channels = 3
    height = 224
    width = 224
    
    model = LowLevelVisionModel()
    x = torch.randn(batch_size, num_frames, channels, height, width)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")