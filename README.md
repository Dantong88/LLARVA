# U2Seg: Unsupervised Universal Image Segmentation

We present **U2Seg**, a unified framework for **U**nsupervised **U**niversal image **Seg**mentation that consistently outperforms previous state-of-the-art methods designed for individual tasks: CutLER for unsupervised instance segmentation, STEGO for unsupervised semantic segmentation, and the naive combination of CutLER and STEGO for unsupervised panoptic segmentation.

<p align="center"> <img src='docs/teaser.jpg' align="center" > </p>

> [**Unsupervised Universal Image Segmentation**](https://u2seg.github.io/)            
> [Dantong Niu*](https://scholar.google.com/citations?user=AzlUrvUAAAAJ&hl=en), [Xudong Wang*](https://people.eecs.berkeley.edu/~xdwang/), [Xinyang Han*](https://xinyanghan.github.io/), [Long Lian](https://tonylian.com/), [Roei Herzig](https://roeiherz.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/)     
> Berkeley AI Research, UC Berkeley    

[comment]: <> (> CVPR 2024    )

[project page](https://llarva24.github.io/) | [arxiv](https://arxiv.org/abs/2312.17243) | [bibtex](#citation)

<a href="https://paperswithcode.com/sota/unsupervised-panoptic-segmentation-on-coco?p=unsupervised-universal-image-segmentation">
  <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-universal-image-segmentation/unsupervised-panoptic-segmentation-on-coco">
</a>
<br>
<a href="https://paperswithcode.com/sota/unsupervised-zero-shot-panoptic-segmentation?p=unsupervised-universal-image-segmentation">
  <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-universal-image-segmentation/unsupervised-zero-shot-panoptic-segmentation">
</a>
<br>
<a href="https://paperswithcode.com/sota/unsupervised-zero-shot-instance-segmentation?p=unsupervised-universal-image-segmentation">
  <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-universal-image-segmentation/unsupervised-zero-shot-instance-segmentation">
</a>
<br>
<a href="https://paperswithcode.com/sota/unsupervised-semantic-segmentation-on-coco-7?p=unsupervised-universal-image-segmentation">
  <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-universal-image-segmentation/unsupervised-semantic-segmentation-on-coco-7">
</a>

<!-- ## Features 
- U2Seg is the first universal unsupervised image segmentation model that can tackle unsupervised semantic-aware instance, semantic and panoptic segmentation tasks using a unified framework.
- U2Seg can learn unsupervised object detectors and instance segmentors solely on ImageNet-1K.
- U2Seg exhibits strong robustness to domain shifts when evaluated on 11 different benchmarks across domains like natural images, video frames, paintings, sketches, etc.
- U2Seg can serve as a pretrained model for fully/semi-supervised detection and segmentation tasks. -->

## Installation
See [installation instructions](INSTALL.md).

## Dataset Preparation
See [Preparing Datasets for U2Seg](datasets/README.md).

## Method Overview
<p align="center">
  <img src="docs/main_pipeline_1.jpg" width=90%>
</p>

U2Seg has 4 stages: 1) generating pseudo instance masks with MaskCut and clustering, 2) generating pseudo semantic masks with STEGO and 3) merging instance masks, semantic masks to get pseudo labels for panoptic segmentation and 4) learning unsupervised universal segmentor from pseudo-masks of unlabeled data.

## Pseudo Mask Gneration
This part includes MaskCut+Clustering, which we use to generate the pseudo for training of U2Seg, additional information will come later. 
For implementers who wants to play with out models, we provide well-processed annotations in Data Preparation.

## Universal Image Segmentation

### Training
After you prepare the dataset following the above instructions, you should be able to train the U2Seg universal segmentation model by:

```
python ./tools/train_net.py  --config-file ./configs/COCO-PanopticSegmentation/u2seg_R50_800.yaml
```
Note: you need to download the pre-trained [dino backbone](https://drive.google.com/file/d/1UtRUgUQK20KS8MGebCWgLPHxrez7mfV4/view?usp=sharing) and change the path of the corresponding ```yaml``` file.

To train U2Seg model with different clustering number (e.g. 300), you can use `configs/COCO-PanopticSegmentation/u2seg_R50_300.yaml` config file and set the environment variable by `export CLUSTER_NUM=300`. (This variable would be used in `detectron2/data/datasets/builtin.py` and `detectron2/data/datasets/builtin_meta.py`)

### Inference
We provide models trained with different cluster numbers and training sets. Each cell in the table below contains a link to the corresponding model checkpoint. Place the downloaded ckpts under `ckpts` folder.

<table>
<thead>
<tr>
<th align="center">Cluster Num</th>
<th align="center">ImageNet</th>
<th align="center">COCO</th>
<th align="center">ImageNet + COCO</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">800</td>
<td align="center"><a href="https://drive.google.com/drive/folders/186GBbIhEW7W0eidGOGRTmTyM_HedSOQh">CheckPoint</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/186GBbIhEW7W0eidGOGRTmTyM_HedSOQh">CheckPoint</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/186GBbIhEW7W0eidGOGRTmTyM_HedSOQh">CheckPoint</a></td>
</tr>
<tr>
<td align="center">300</td>
<td align="center"><a href="https://drive.google.com/drive/folders/186GBbIhEW7W0eidGOGRTmTyM_HedSOQh">CheckPoint</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/186GBbIhEW7W0eidGOGRTmTyM_HedSOQh">CheckPoint</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/186GBbIhEW7W0eidGOGRTmTyM_HedSOQh">CheckPoint</a></td>
</tr>
</tbody>
</table>  
 

To run inference on images, you should first assign a checkpoint in the ```u2seg_eval.yaml```, then:
```
python ./demo/u2seg_demo.py --config-file configs/COCO-PanopticSegmentation/u2seg_eval_800.yaml --input demo/images/*jpg --output results/demo_800 
```

To test model trained with different clustering number (e.g. 300), you can use config file like this `configs/COCO-PanopticSegmentation/u2seg_R50_300.yaml`.

We give a few demo images in ```demo/images``` and the corresponding visualizations of the panoptic segmentation with U2Seg:
<p align="center">
  <img src="docs/u2seg-demo.png" width=80%>
</p>


### Evaluation
To reproduce the quantitative evaluation results in the main paper, one can following the steps below.

#### 1. Prepare the ground truth annotations
Since the original official panoptic annotations have 53 semantic categories, while u2seg has 15 supercategories, first run [```get_panoptic_anns_supercategory.py```](https://github.com/u2seg/U2Seg/blob/main/datasets/prepare_ours/get_panoptic_anns_supercategory.py) to generate the supercategory version of groudtruth, then you will get ```panoptic_val2017_300super.json``` under the ```detectron2/datasets/datasets/panoptic_anns``` path.

#### 2. Set the CATEGORIES
User should manually comment out line 42-176 in [```detectron2/data/datasets/builtin_meta.py```](https://github.com/u2seg/U2Seg/blob/148cc100c3a13426700af619aad68a6afa5c5f6f/detectron2/data/datasets/builtin_meta.py#L42) when switch from training to evaluation. Also user needs to maunally change the ```cluster_num``` in line 39 in [```detectron2/evaluation/panoptic_evaluation.py```](https://github.com/u2seg/U2Seg/blob/148cc100c3a13426700af619aad68a6afa5c5f6f/detectron2/evaluation/panoptic_evaluation.py#L39) based on your own seeting.

#### 3. Get hungarain mapping dict
We do the hungarian matching to map our cluter id to real cagetory id. The user needs to run:

```
export CLUSTER_NUM=300
python tools/train_net.py --config-file configs/COCO-PanopticSegmentation/u2seg_eval_300.yaml --eval-only --eval-mode hungarian_matching --num-gpus 1
```
* note that this should be run only by using one single gpu. After that, a folder named ```hungarian_matching``` should be gnerated under ```detectron2/```, we provide an [hungarian mapping dict example with cluter number is 300](https://drive.google.com/drive/folders/19ka73SH1mHAfflzlZpA2McWeZakLujEX?usp=sharing) the user can have a quick check of the intermediate results.

#### 4. Get evaluation results for instance segmentation, semantic segmentation and panoptic segmentation
```
export CLUSTER_NUM=300
python tools/train_net.py --config-file configs/COCO-PanopticSegmentation/u2seg_eval_300.yaml --eval-only --eval-mode eval --num-gpus 8
```
* this can be executed by multiple gpus.




### Efficient Learning
We give the reproduction of our efficient learning part in [U2Seg_eff](https://github.com/Dantong88/U2Seg_eff). See that repo if you want to play with this part.

## License
U2Seg is licensed under the [Apache](LICENSE), however portions of the project are available under separate license terms: CutLER, Detectron2 and DINO are licensed under the CC-BY-NC license; TokenCut, Bilateral Solver and CRF are licensed under the MIT license; If you later add other third party code, please keep this license info updated, and please let us know if that component is licensed under something other than CC-BY-NC, MIT, or CC0.

## Ethical Considerations
U2Seg's wide range of detection capabilities may introduce similar challenges to many other visual recognition methods.
As the image can contain arbitrary instances, it may impact the model output.

## How to get support from us?
If you have any general questions, feel free to email us at [Dantong Niu](mailto:bias_88@berkeley.edu), [Xudong Wang](mailto:xdwang@eecs.berkeley.edu), [Xinyang Han](mailto:hanxinyang66@gmail.com). If you have code or implementation-related questions, please feel free to send emails to us or open an issue in this codebase (We recommend that you open an issue in this codebase, because your questions may help others). 

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@misc{niu2023unsupervised,
      title={Unsupervised Universal Image Segmentation}, 
      author={Dantong Niu and Xudong Wang and Xinyang Han and Long Lian and Roei Herzig and Trevor Darrell},
      year={2023},
      eprint={2312.17243},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- ## Features 
- U2Seg is the first universal unsupervised image segmentation model that can tackle unsupervised semantic-aware instance, semantic and panoptic segmentation tasks using a unified framework.
- U2Seg can learn unsupervised object detectors and instance segmentors solely on ImageNet-1K.
- U2Seg exhibits strong robustness to domain shifts when evaluated on 11 different benchmarks across domains like natural images, video frames, paintings, sketches, etc.
- U2Seg can serve as a pretrained model for fully/semi-supervised detection and segmentation tasks. -->



