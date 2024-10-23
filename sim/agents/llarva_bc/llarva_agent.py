import copy
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from yarr.agents.agent import Agent, Summary, ActResult, \
    ScalarSummary, HistogramSummary

from helpers import utils
from PIL import Image
import re
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

class Actor(nn.Module):

    def __init__(self, actor_network: nn.Module):
        super(Actor, self).__init__()
        self._actor_network = copy.deepcopy(actor_network)
        self._actor_network.build()

    def forward(self, observations, robot_state, lang_goal_emb):
        mu = self._actor_network(observations, robot_state, lang_goal_emb)
        return mu


class LLaRVAAgent(Agent):

    def __init__(self,
                 policy_ckpt,
                 temperature = 0,
                 with_visual_trace = False
                 ):
        """Constructor."""
        # Model
        self.conv_mode = 'llava_v1'
        self.temperature = temperature
        self.top_p = None
        self.num_beams = 1
        disable_torch_init()
        model_path = os.path.expanduser(policy_ckpt)
        self.model_name = get_model_name_from_path(model_path)
        self.policy_ckpt = policy_ckpt
        self.with_visual_trace = False


    def build(self, training: bool, device: torch.device = None):
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.policy_ckpt, None,
                                                                                                   self.model_name,
                                                                                                   device=device)
        self.model.eval()
        self.device = device

    def reset(self) -> None:
        self.prev_actions_buffer = [[0, 0, 0, 0, 0, 0, 0]] * 5
        self.rollout_step_counter = 0
        self.tj = {}


    def normalize_and_add_gaussian(self, action):
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, action.shape)
        noisy_action = action + noise

        return noisy_action

    def act(self, step: int, observation: dict, task_goal: str,
            deterministic=False) -> ActResult:

        """Step function."""
        # Language
        instruction = task_goal
        # print(instruction)
        robot_type = 'Franka'
        control_type = 'joint control'
        pred_num_step = 1

        # processing the action
        # update the prev action
        current_pose = observation['joint_pos'][-1]
        add_noise = True
        if add_noise:
            current_pose = self.normalize_and_add_gaussian(current_pose)
        current_pose = np.round(current_pose, 4).tolist()

        self.prev_actions_buffer.append(current_pose)
        self.prev_actions_buffer = self.prev_actions_buffer[1:]

        if self.with_visual_trace:
            qs = 'You are a {} using the {}. The task is \"{}\", and the previous five (including current) steps are {}. Can you predict the 2D visual trace of the end effector and the action of the next 1 step?'.format(
                robot_type, control_type, instruction, self.prev_actions_buffer, pred_num_step)
        else:
            qs = 'You are a {} robot using the {}. The task is \"{}\", and the previous five (including current) steps is {}, can you predict action of the next {} step?'.format(
            robot_type, control_type, instruction, self.prev_actions_buffer, pred_num_step)

        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(self.device)

        image = np.transpose(observation['front_rgb'][-1],(1,2,0))
        image = Image.fromarray(image)
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]

        debug = True # if you want save the vis
        if debug:
            save_path = '/home/niudt/tmp/release/sweep/{}.jpg'.format(self.rollout_step_counter)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                image_sizes=[image.size],
                do_sample=True if self.temperature > 0 else False,
                temperature=None,
                top_p=self.top_p,
                num_beams=self.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        self.tj[self.rollout_step_counter] = outputs
        if not self.with_visual_trace:
            # Extracting the list of numbers using regular expression
            numbers_str = re.search(r'\[(.*?)\]', outputs).group(1)

            # Splitting the numbers string into individual numbers
            # here add rectify for if it find multiple '[
            numbers = [self.clean_string(x) for x in numbers_str.split(',')]
            if len(numbers) == 7:
                numbers.append(1.0)
            numbers_list = [float(x) for x in numbers]
            numpy_array = np.array(numbers_list)
        else:
            match = re.search(r"The next action step: \[([^\]]+)\]", outputs)
            if match:
                # Extract the string and split it into a list of strings
                action_step_str = match.group(1)
                action_step_list = action_step_str.split(", ")

                # Convert the list of strings to a list of floats
                action_step_floats = [float(x) for x in action_step_list]

                # Convert the list of floats to a NumPy array
                numpy_array = np.array(action_step_floats)

                # print(action_step_array)
            else:
                print("No match found")

        action_pred = numpy_array

        self.rollout_step_counter += 1

        print('step {} pred action: {}'.format(step, action_pred))

        return ActResult(action=action_pred)

    def clean_string(self, input_string):
        # Use regex to remove unwanted characters
        cleaned_string = re.sub(r'[^0-9\-.]', '', input_string)
        return cleaned_string

    def save_tj(self, path):
        import json

        # Write the dictionary to a JSON file
        with open(path, 'w') as json_file:
            json.dump(self.tj, json_file)
            print('save the json')

    def update_summaries(self) -> List[Summary]:
        return []

    def update(self, step: int, replay_sample: dict) -> dict:
        return []

    def act_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str):
        pass

    def save_weights(self, savedir: str):
        pass
