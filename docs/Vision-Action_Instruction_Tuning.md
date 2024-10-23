## Vision-Action Instruction Tuning

When used the pre-training model in a specific downstream task, i.e. robot manipulation tasks, we need to further 
do instruction tuning using collected demonstrations in the new environment setting. In this repo, we provide instructions to reproduce the 
application in [RLBench Benchmark](https://github.com/stepjam/RLBench).

Our code is built based mainly on [PerAct](https://github.com/peract/peract) and [RLBench](https://github.com/stepjam/RLBench), make sure you might need to
cite them if find it useful. 

The following content including 4 parts:
* Simulation Environment Installation
* Prepare Simulation Data
* Instruction Tuning
* Inference

***
### Simulation Environment Installation
#### 1. You can still use the ``llarva`` set in pre-training.
```angular2html
conda activate llarva
```
#### 2. PyRep and Coppelia Simulator

Follow instructions from the official [PyRep](https://github.com/stepjam/PyRep) repo; reproduced here for convenience:

PyRep requires version **4.1** of CoppeliaSim. Download: 
- [Ubuntu 16.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu16_04.tar.xz)
- [Ubuntu 18.04](https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Player_V4_1_0_Ubuntu18_04.tar.xz)
- [Ubuntu 20.04](https://www.coppeliarobotics.com/previousVersions#)

Once you have downloaded CoppeliaSim, you can pull PyRep from git:

```bash
cd LLARVA/sim
git clone https://github.com/stepjam/PyRep.git
cd PyRep
```

Add the following to your *~/.bashrc* file: (__NOTE__: the 'EDIT ME' in the first line)

```bash
export COPPELIASIM_ROOT=<EDIT ME>/PATH/TO/COPPELIASIM/INSTALL/DIR
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

Remember to source your bashrc (`source ~/.bashrc`) or 
zshrc (`source ~/.zshrc`) after this.

**Warning**: CoppeliaSim might cause conflicts with ROS workspaces. 

Finally install the python library:

```bash
pip install -r requirements.txt
pip install .
```

You should be good to go!
You could try running one of the examples in the *examples/* folder.

If you encounter errors, please use the [PyRep issue tracker](https://github.com/stepjam/PyRep/issues).

#### 3. RLBench

LLARVA uses my [RLBench fork](https://github.com/Dantong88/RLBench/tree/LLARVA). 

```bash
cd LLARVA/sim
git clone -b LLARVA https://github.com/Dantong88/RLBench # note: 'LLARVA' branch

cd RLBench
pip install -r requirements.txt
python setup.py develop
```

For [running in headless mode](https://github.com/MohitShridhar/RLBench/tree/peract#running-headless), tasks setups, and other issues, please refer to the [official repo](https://github.com/stepjam/RLBench).

#### 4. YARR

LLARVA uses my [YARR fork](https://github.com/Dantong88/YARR/tree/LLARVA).

```bash
cd LLARVA/sim
git clone -b LLARVA https://github.com/Dantong88/YARR # note: 'LLARVA' branch

cd YARR
pip install -r requirements.txt
python setup.py develop
```

#### 5. LLARVA-sim
```bash
cd LLARVA/sim
pip install -r requirements.txt
python setup.py develop
```

***

### Prepare Simulation Data

Then you need to generate the simulation demonstrations, which are used in our vision-action instruction
tuning.

#### 1. Generate RLBench Demos
```angular2html
cd LLARVA/sim/RLBench/tools
export SIM_ROOT=LLARVA/sim
python dataset_generator.py --tasks=sweep_to_dustpan_of_size \
                            --save_path=$SIM_ROOT/data/val \
                            --image_size=128,128 \
                            --renderer=opengl \
                            --episodes_per_task=10 \
                            --processes=1 \
                            --all_variations=True
```

After this, you will generate 10 demos in ``$SIM_ROOT/data/val``, you maybe need to generate more demos using in training.

#### 2. Adapt the Format (Skip if you only want to test the model)

Before using the demos to do vision-action tuning, you should first transfer the format to get annotations.
We provide script as follows:
```angular2html
cd /LLARVA/sim
python generate_vision-action-tuning_anns.py --data-path data/val --save-path data/anns/train.json --selected-task sweep_to_dustpan_of_size
```
You will get ``train.json`` in ``data/anns`` folder, which can further be used in vision-action instruction tuning.

We also provide the one example of the generated demos [800 sweep_to_dustpan_of_size](https://drive.google.com/file/d/14EbwEPJwmqjKNpoPr6rXiv_2W1J2DkNu/view?usp=sharing) and its processed annotations[annotations](https://drive.google.com/drive/folders/1JQouifNi3sZMMYolE4Oqp8fR6H78sdlf?usp=sharing), including two version: with/without
visual traces, you can download and put them with the following structure:

```angular2html
LLARVA/sim
│ 
└── data
│   ├── anns
│   │   ├── sweep_to_dustpan_of_size
│   │       └── (our pre-processed vision-action annotations instruction annotations, including both with/without visual trace)
│   │
│   └── val
│   │   └── (10 demos preciously generated)
│   │
│   └── sweep_to_dustpan_of_size 
│       └── (demos download by link)         
│
└── generate_vision-action-tuning_anns.py
    ... 
```

***

### Instruction Tuning
After get the training annotations, you can follow the following steps to adapt the pre-trained model to specific downstream task.

#### 1. Put the Pre-training Weight
(download in [Vision-Action Instruction Pre-training.md](https://github.com/Dantong88/LLARVA/blob/main/docs/Vision-Action_Instruction_Pre-training.md)) in the output folder, for example:
```angular2html
cd LLARVA/sim
mkdir output
cd output
mkdir llava-lora-instruction-tuning-sweep_to_dustpan_of_size
cp -r the/path/pretrained_model  llava-lora-instruction-tuning-sweep_to_dustpan_of_size
```

####  2. Fix the Package Incompatibilities.

We find some incompatibilities of the ``deepspeed`` and ``transformers``
package with our fine-tuning code, so you need to go to your conda llarva environment (common in ``anaconda3/envs/llarva/lib/python3.10/sitpackages/``), then manually modify source package as follows:

* Set ``load_module_strict = False, load_optimizer_states=False, load_lr_scheduler_states=False`` in [transformers/integrations/deepspeed.py](https://github.com/huggingface/transformers/blob/21d5025826857e11a75ef7b23ac15a607be4fc54/src/transformers/integrations/deepspeed.py#L438)
* Comment out the whole ``try-except`` block in [transformers/generation/configuration_utils.py](https://github.com/huggingface/transformers/blob/32590b5ecb50f1c56be32cb0e686196be9427f2f/src/transformers/generation/configuration_utils.py#L827)
* Set ``self._load_optimizer_and_scheduler(None)`` at [transformers/trainer.py](https://github.com/huggingface/transformers/blob/21d5025826857e11a75ef7b23ac15a607be4fc54/src/transformers/trainer.py#L2295), and add ``resume_from_checkpoint = None`` after that.



#### 3. Launch the instruction tuning.

```angular2html
cd LLARVA
bash scripts/v1_5/vision-action_instruction_tuning_rlbench.sh
```


#### 4. Merge the Lora Weight

After training, you should first merge the lora weights by running:
```angular2html
python scripts/merge_lora_weights.py --model-path 'path/to/your/lora-weights' --model-base 'lmsys/vicuna-7b-v1.5' --save-model-path 'your/path'
```
*Note that your ``save-model-path`` should include word ``llava``, otherwise, you might get error.*


We release our weights for "meat off grill" task in RLBench as follows (this is the merged final weights, i.e. you do not need to run the above command to merge it before using it in inference):

#### Lora Weight
<!--
./gen_html_table.py --config 'COCO-Detection/retina*50*' 'COCO-Detection/retina*101*' --name R50 R50 R101 --fields lr_sched train_speed inference_speed mem box_AP
-->

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Size</th>
<th valign="bottom">Tuning Set</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: retinanet_R_50_FPN_1x -->
<tr><td align="left">LLARVA</td>
<td align="center">7B</td>
<td align="center">800 episodes of sweep_to_dustpan_of_size</th> in RLBench</td>
<td align="center">Vicuna-7B</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1uwCIFQ20a4yk5yOE073d82gELjM7_QaS?usp=sharing">Model</a></td>
</tr>
</tbody></table>

***
### Inference

To test the model on RLBench, just run

```angular2html
cd LLRVA/sim
export SIM_ROOT=LLRVA/sim
python eval.py \
rlbench.tasks=[sweep_to_dustpan_of_size] \
rlbench.demo_path=$SIM_ROOT/data/val \
framework.eval_from_eps_number=0 \
framework.eval_episodes=10 \
rlbench.episode_length=120 \
framework.gpu=4 \
method.ckpt=path/to/llava-sweep_to_dustpan_of_size_merged
```


### Citations
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
#### LLARVA
```
@misc{niu2024llarva,
      title={LLARVA: Vision-Action Instruction Tuning Enhances Robot Learning}, 
      author={Dantong Niu and Yuvan Sharma and Giscard Biamby and Jerome Quenum and Yutong Bai and Baifeng Shi and Trevor Darrell and Roei Herzig},
      year={2024}
}
```

#### PerAct

```
@inproceedings{shridhar2022peract,
  title     = {Perceiver-Actor: A Multi-Task Transformer for Robotic Manipulation},
  author    = {Shridhar, Mohit and Manuelli, Lucas and Fox, Dieter},
  booktitle = {Proceedings of the 6th Conference on Robot Learning (CoRL)},
  year      = {2022},
}
```

#### RLBENCH
```
@article{james2020rlbench,
  title={Rlbench: The robot learning benchmark \& learning environment},
  author={James, Stephen and Ma, Zicong and Arrojo, David Rovick and Davison, Andrew J},
  journal={IEEE Robotics and Automation Letters},
  volume={5},
  number={2},
  pages={3019--3026},
  year={2020},
  publisher={IEEE}
}
```




