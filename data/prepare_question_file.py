import os
import json
import random
from tqdm import tqdm



if __name__ == '__main__':
    # val_file = json.load(open('data/annotations/val.json'))
    val_file = json.load(open('/scratch/partial_datasets/rlbench/release_oxe/annotations/train.json'))

    episode = {}


    our_questionset = []
    random.shuffle(val_file)
    for i, sample in enumerate(tqdm(val_file[:100])):
        current_question = {}
        current_question['image'] = sample['image']
        # filter the episode
        current_episode = sample['image'].split('/')[-2]
        current_episode_index = sample['image'].split('/')[-1].split('_')[0]


        current_question['question_id'] = i
        current_question['category'] = 'conv'

        # for question
        for i, conv in enumerate(sample['conversations']):
            if conv['from'] == 'human':
                text = conv['value'].split('<image>\n')[1]
                current_question['text'] = text

            else:
                text = conv['value']
                current_question['gt'] = text
        our_questionset.append(current_question)


    # Path to your JSONL file
    file_path = 'data/test_questions.jsonl'

    # Write the dictionaries to a JSONL file
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as file:
        for d in our_questionset:
            json_line = json.dumps(d) + '\n'  # Convert dict to JSON string and add newline
            file.write(json_line)

