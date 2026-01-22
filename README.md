# Beyond Visual Safety: Jailbreaking Multimodal Large Language Models for Harmful Image Generation via Semantic-Agnostic Inputs

This folder contains the official implementation of the **BVS** jailbreak framework. By utilizing the **MIDOS** (Minimal Inverse-Distance Optimal Selection) algorithm, this tool automatically transforms a text prompt into a fragmented inducing image $I_S$ designed to explore the visual safety boundaries of MLLMs.

## Project Structure

Please maintain the following directory structure within the `BVS` folder:

```text
BVS/
├── ISGen.py                # The main execution script
├── requirements.txt        # Python dependencies
├── clip-ViT-B-32/          # Local semantic embedding model
├── CogView4/               # Local T2I model (CogView4-6B)
├── NImage/                 # Neutralized Image Data (benign reference set)
└── output/                 # Default directory for generated results
```

## Installation

Clone or download this BVS folder.

Navigate to the folder and install the dependencies:

```text
pip install -r requirements.txt
```

## Model Sources

Please download the pre-trained weights from the following official repositories before running the script:

**CogView4 (Image Generation Model):**

We use **CogView4-6B** as the generative backbone. You can download the weights from [Hugging Face](https://huggingface.co/THUDM/cogview4-6b) or [ModelScope](https://modelscope.cn/models/ZhipuAI/cogview4-6b). After downloading, place the folder in the root directory or specify the path using `--gen_model_path`.

 https://huggingface.co/THUDM/cogview4-6b.

**CLIP (Embedding Model):**
For semantic evaluation, we use **clip-vit-base-patch32**. You can either specify the Hugging Face model ID (`openai/clip-vit-base-patch32`) to allow automatic downloading or download the weights manually to `./clip-ViT-B-32`.

https://huggingface.co/openai/clip-vit-base-patch32.



## Usage

You can generate the inducing image $I_S$ directly from a text prompt. The script handles the generation of the initial malicious image ($I_A$), semantic fragmentation, and the final 3x3 matrix splicing.

```text
python ISGen.py \
    --prompt "A detailed description of the prohibited content" \
    --gen_model_path "./CogView4" \
    --embed_model_path "./clip-ViT-B-32" \
    --neutral_dir "./NImage" \
    --output_dir "./output"
```

