## Vision-Action Instruction Pre-training

For the vision-action instruction pre-training, the implementation of LLARVA is built based on [LLaVA](https://github.com/haotian-liu/LLaVA).

### Installation
1. Clone this repository
```bash
git clone https://github.com/Dantong88/LLARVA
cd LLAVA
```

2. Install Package
```Shell
conda create -n llarva python=3.10 -y
conda activate llarva
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

### Prepare the Data

1. See [DATASET.md]() to download both image.zip file and annotations, put them under ./data folder.
2. Download off-the-shelf projector weights:
```angular2html
git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
```
