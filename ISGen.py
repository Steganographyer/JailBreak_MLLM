import torch
import os
import argparse
import random
import numpy as np
from PIL import Image
from diffusers import CogView4Pipeline
from sentence_transformers import SentenceTransformer, util

class BVSPipeline:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 初始化 CogView4 生成模型
        print(f"[*] 正在加载生成模型: {args.gen_model_path}")
        self.pipe = CogView4Pipeline.from_pretrained(
            args.gen_model_path, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        # 2. 初始化语义提取模型 (MIDOS)
        print(f"[*] 正在加载语义模型: {args.embed_model_path}")
        self.embed_model = SentenceTransformer(args.embed_model_path, device=self.device)

    def generate_malicious_image(self, prompt):
        """生成原始有害图像 I_A"""
        print(f"[*] 正在生成有害图像 $I_A$...")
        image = self.pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_inference_steps=50,
            width=1024,
            height=1024,
        ).images[0]
        temp_path = os.path.join(self.args.output_dir, "temp_IA.png")
        image.save(temp_path)
        return image, temp_path

    def get_embedding(self, pil_image):
        """提取图像的语义特征向量"""
        return self.embed_model.encode(pil_image, convert_to_tensor=True, show_progress_bar=False)

    def calculate_distance(self, feat1, feat2):
        """计算语义距离 (1 - Cosine Similarity)"""
        sim = util.cos_sim(feat1, feat2).item()
        return max(1e-6, 1.0 - sim)

    def prepare_fragmented_quadrants(self, img_a, size=200):
        """将 I_A 拆分为打乱的 4 个碎片 (b11, b12, b21, b22)"""
        img = img_a.resize((size, size), Image.Resampling.LANCZOS)
        half = size // 2
        raw_quadrants = [
            img.crop((0, 0, half, half)),    # Q1
            img.crop((half, 0, size, half)), # Q2
            img.crop((0, half, half, size)), # Q3
            img.crop((half, half, size, size))# Q4
        ]
        random.shuffle(raw_quadrants)
        return raw_quadrants

    def load_neutralized_pool(self, folder_path, patch_size=100):
        """加载 Neutralized Image Data (备选库) 并预计算特征"""
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        candidates = []
        print(f"[*] 正在预处理 Neutralized 数据库 (共 {len(files)} 张)...")
        for f in files:
            path = os.path.join(folder_path, f)
            try:
                img = Image.open(path).convert("RGB")
                img_resized = img.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
                emb = self.get_embedding(img_resized)
                candidates.append({"img": img_resized, "emb": emb})
            except Exception as e:
                print(f"[!] 跳过损坏图片 {f}: {e}")
        return candidates

    def midos_selection(self, img_a, quadrants_data, candidates):
        """MIDOS 核心逻辑：基于语义距离选择 a1-a5 以稀释有害内容"""
        emb_a = self.get_embedding(img_a)
        emb_b = [q["emb"] for q in quadrants_data] # b11, b12, b21, b22

        used_indices = set()
        selection = {}

        def get_inverse_dist_sq_score(cand_emb, base_embs):
            return sum([1.0 / (self.calculate_distance(cand_emb, b_emb) ** 2) for b_emb in base_embs])

        # 选择 a3: 使与 I_A 的语义距离最大 (最大化稀释)
        dist_scores = [self.calculate_distance(emb_a, c["emb"]) for c in candidates]
        a3_idx = np.argmax(dist_scores)
        selection['a3'] = candidates[a3_idx]['img']
        used_indices.add(a3_idx)

        # 依次选择相邻块 a1, a2, a4, a5
        tasks = [
            ('a1', [emb_b[0], emb_b[1]]), # 衔接 b11, b12
            ('a2', [emb_b[0], emb_b[2]]), # 衔接 b11, b21
            ('a4', [emb_b[1], emb_b[3]]), # 衔接 b12, b22
            ('a5', [emb_b[2], emb_b[3]])  # 衔接 b21, b22
        ]

        for key, bases in tasks:
            min_score = float('inf')
            best_idx = -1
            for i, c in enumerate(candidates):
                if i in used_indices: continue
                score = get_inverse_dist_sq_score(c["emb"], bases)
                if score < min_score:
                    min_score = score
                    best_idx = i
            selection[key] = candidates[best_idx]['img']
            used_indices.add(best_idx)
        return selection

    def assemble_is(self, quadrants_data, selection, patch_size=100):
        """矩阵拼接：生成最终的诱导图 I_S (3x3 矩阵)"""
        result = Image.new("RGB", (patch_size * 3, patch_size * 3))
        # 第一行: b11, a1, b12
        result.paste(quadrants_data[0]['img'], (0, 0))
        result.paste(selection['a1'], (patch_size, 0))
        result.paste(quadrants_data[1]['img'], (2 * patch_size, 0))
        # 第二行: a2, a3, a4
        result.paste(selection['a2'], (0, patch_size))
        result.paste(selection['a3'], (patch_size, patch_size))
        result.paste(selection['a4'], (2 * patch_size, patch_size))
        # 第三行: b21, a5, b22
        result.paste(quadrants_data[2]['img'], (0, 2 * patch_size))
        result.paste(selection['a5'], (patch_size, 2 * patch_size))
        result.paste(quadrants_data[3]['img'], (2 * patch_size, 2 * patch_size))
        
        output_path = os.path.join(self.args.output_dir, "final_IS.png")
        result.save(output_path, "PNG")
        print(f"[*] ✅ 诱导图 $I_S$ 已生成并保存至: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="BVS Framework: From Prompt to Inducing Image IS")
    parser.add_argument("--prompt", type=str, required=True, help="有害内容的初始提示词")
    parser.add_argument("--gen_model_path", type=str, default="THUDM/CogView4-6B", help="CogView4 模型路径")
    parser.add_argument("--embed_model_path", type=str, required=True, help="SentenceTransformer/CLIP 模型本地路径")
    parser.add_argument("--neutral_dir", type=str, required=True, help="Neutralized Image Data (无害图库) 文件夹路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="结果保存目录")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 运行 Pipeline
    bvs = BVSPipeline(args)
    
    # 1. 生成有害图 IA
    img_a, _ = bvs.generate_malicious_image(args.prompt)
    
    # 2. 准备 IA 碎片
    b_imgs = bvs.prepare_fragmented_quadrants(img_a)
    quadrants_data = [{"img": img, "emb": bvs.get_embedding(img)} for img in b_imgs]
    
    # 3. 加载无害图池
    candidates = bvs.load_neutralized_pool(args.neutral_dir)
    
    # 4. 执行 MIDOS 选择并拼接
    print("[*] 正在执行 MIDOS 语义稀释算法...")
    selection = bvs.midos_selection(img_a, quadrants_data, candidates)
    bvs.assemble_is(quadrants_data, selection)

if __name__ == "__main__":
    main()