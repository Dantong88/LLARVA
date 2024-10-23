import pickle
import numpy as np
import json
import os
from tqdm import tqdm
import argparse

def get_curr_action(item):
    joint_velocities = getattr(item, "joint_velocities")
    # inverting gripper since our convention is gripper as 1 = closed gripper
    # gripper_open = 1 if getattr(item, "gripper_open") == 0 else 0
    gripper_open = getattr(item, "gripper_open")
    temp_arr = np.append(joint_velocities, gripper_open)
    return np.round(temp_arr, 4).tolist()


def get_curr_dict(total_count, image_path, prev_5_positions, curr_action, variation_des):
    conversations = [{
                         "value": f"<image>\nYou are a Franka robot using the joint control. The task is \"{variation_des}\", and the previous five (including current) steps are {prev_5_positions}. Can you predict action of the next 1 step?",
                         "from": "human"}, {"from": "gpt", "value": f"The next action step: {curr_action}"}]
    curr_dict = {"id": total_count, "image": image_path, "conversations": conversations}
    return curr_dict


def apply_noise(position):
    noise = np.random.normal(0, 0.01, position.shape)
    noisy_position = position + noise
    return noisy_position


def generate(args):
    selected_task = args.selected_task

    for task in selected_task:
        simulation_path = "{}/{}/all_variations/episodes".format(args.data_path, task)
        final_json_content = []
        total_count = 0
        camera_view = 'front_rgb'

        for episode in tqdm(sorted(os.listdir(simulation_path))):
            current_episode_path = os.path.join(simulation_path, episode)

            variation_des_pickle = os.path.join(current_episode_path, "variation_descriptions.pkl")
            with open(variation_des_pickle, 'rb') as file:
                variation_des = pickle.load(file)

            pickle_path = os.path.join(current_episode_path, "low_dim_obs.pkl")

            add_gripper_joint_pos = False

            if add_gripper_joint_pos:
                prev_5_positions = [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * 5
            else:
                prev_5_positions = [[0, 0, 0, 0, 0, 0, 0]] * 5
            with open(pickle_path, 'rb') as file:
                data = pickle.load(file)
                counter = 0
                for item_idx, item in enumerate(data):
                    try:
                        curr_action = get_curr_action(data[item_idx + 1])
                    except:
                        continue
                    curr_position = getattr(item, "joint_positions")

                    if add_gripper_joint_pos:
                        gripper_joint_pos = getattr(item, "gripper_joint_positions")
                        curr_position = np.concatenate([curr_position, gripper_joint_pos])

                    add_noise = True
                    if add_noise:
                        curr_position = np.round(apply_noise(curr_position), 4).tolist()

                    else:
                        curr_position = np.round(curr_position, 4).tolist()
                    prev_5_positions.append(curr_position)
                    prev_5_positions.pop(0)

                    image_path = f"{episode}/{camera_view}/{counter}.png"
                    final_json_content.append(
                        get_curr_dict(total_count, image_path, prev_5_positions, curr_action, variation_des[0]))
                    print(variation_des[0])

                    counter += 1
                    total_count += 1
                    # sys.exit(0)

        import random
        random.shuffle(final_json_content)

        output_path = args.save_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as json_file:
            json.dump(final_json_content, json_file)

        print(total_count)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/niudt/project/LLARVA/sim/data/val")
    parser.add_argument("--save-path", type=str, default='/home/niudt/project/LLARVA/sim/data/anns/train.json')
    parser.add_argument("--selected-task", type=str, nargs='+', default=['sweep_to_dustpan_of_size'])
    args = parser.parse_args()

    generate(args)





