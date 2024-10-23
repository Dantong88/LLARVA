## Vision-Action Instruction Pre-training

For the vision-action instruction pre-training, the implementation of LLARVA is built based on [LLaVA](https://github.com/haotian-liu/LLaVA).

### Installation
1. Clone this repository
```bash
git clone https://github.com/Dantong88/LLARVA
cd LLARVA
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
2. Download off-the-shelf projector weights, and put it under the LLARVA
```angular2html
cd LLARVA
git clone https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
```

### Launch the Pre-training
```angular2html
bash scripts/v1_5/vision-action_instruction_pretraining_lora.sh
```

### Inference/Demos
1. Merge the lora weights. 

After training, you should first merge the lora weights by running:
```angular2html
python scripts/merge_lora_weights.py --model-path 'path/to/your/lora-weights' --model-base 'lmsys/vicuna-7b-v1.5' --save-model-path 'your/path'
```
*Note that your ``save-model-path`` should include word ``llava``, otherwise, you might get error.*


We release our weights as follows (this is the un-merged lora weights, i.e. you need to run the above command to merge it before using it in inference):

#### Lora Weight
<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50 R50 R101 --fields lr_sched train_speed inference_speed mem box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Size</th>
<th valign="bottom">Train Set</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_FPN_1x -->
<tr><td align="left">LLARVA</td>
<td align="center">7B</td>
<td align="center"><a href="https://github.com/Dantong88/LLARVA/blob/main/docs/DATASET.md">OXE Vision-Action Instruction Pre-training Dataset</a></td>
<td align="center">Vicuna-7B</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BOWZn-jFdLLzutXWZmdit3cDv8qXezs8?usp=sharing">Model</a></td>
</tr>
</tbody></table>

2. Run inference.

First, you should generate your question file, for questioning the model. We provide the smaller [question examples sets](https://drive.google.com/file/d/1O9jFAgy9wzoOVSs3x2Uox3gJb9F3I6Rz/view?usp=sharing) including
100 qestions, you can download it and put it in data folder.

You can easily test the model by running:


```angular2html
python llava/eval/model_vqa.py --model-path 'path/to/merged/model' --image-folder 'data'  --question-file 'data/test_questions.jsonl' --answers-file 'data/answers.jsonl'
```

The results are saved as jsonl file, you can simply read or plot the predicted visual trace by reading the jsonl file.

