import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def extract_features(image_dir, output_dir, model_name="laion/CLIP-ViT-B-32-laion2B-s34B-b79K"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 检查GPU可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载CLIP模型和处理器
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    # 获取所有jpg文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

    # 处理每个图像
    for image_file in tqdm(image_files, desc="处理图像"):
        image_path = os.path.join(image_dir, image_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}.json")

        # 读取和预处理图像
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt").to(device)

        # 提取特征
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # 将特征转换为Python列表
        features_list = image_features.cpu().numpy().tolist()[0]

        # 保存为JSON
        with open(output_file, 'w') as f:
            json.dump({"file_name": image_file, "features": features_list}, f)

        print(f"处理完成: {image_file}")

    print("所有图像处理完成")

# 使用函数
extract_features('222', 'clip')