import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default='/home/niudt/llava-v1.5-7b-lora_exp5_fine-tuning_April29', required=False)
    # parser.add_argument("--model-base", type=str, required=False, default='lmsys/vicuna-7b-v1.5')
    # parser.add_argument("--save-model-path", type=str, required=False, default='/home/niudt/tmp/llarva_release_test')
    #
    # args = parser.parse_args()
    #
    # merge_lora(args)
