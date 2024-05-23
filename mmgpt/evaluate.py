import argparse
import random

import numpy as np
import os
import pandas as pd
import torch
from mmengine import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from mmgpt import create_model_and_transforms
from mmgpt.datasets import InfiniteSampler, build_dataset
from mmgpt.train.distributed import init_distributed_device


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="checkpoints/llama-7b_hf", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenizer_path",
        default="checkpoints/llama-7b_hf",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--pretrained_path",
        default="checkpoints/OpenFlamingo-9B/checkpoint.pt",
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="train-my-gpt4",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    # data args
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tuning_config", type=str, default=None, help="path to tuning config file")
    parser.add_argument("--dataset_config", type=str, default=None, help="path to dataset config file")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    args = parser.parse_args()
    return args

def eval_model(args, model, test_dataloader, lang_dataloader, device_id):
    model.eval()
    for num_steps, batch in tqdm(enumerate(test_dataloader), disable=args.rank != 0):
        #### VISION FORWARD PASS ####
        images = batch["image"].to(device_id).unsqueeze(1).unsqueeze(1)
        input_ids = batch["input_ids"].to(device_id)
        attention_mask = batch["attention_mask"].to(device_id)
        labels = batch["labels"].to(device_id)

        all_outputs = []
        all_outputs_decoded = []
        all_labels = []

        with torch.no_grad():
            outputs = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                # labels=labels,
            )
            output_ids = model.module.generate(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                max_length=1024,
                num_beams=5,
                temperature=1.0,
                top_k=50,
                top_p= 0.95,
                do_sample=True,
            )
            outputs_decoded = model.tokenizer.decode(output_ids, skip_special_tokens=True)
            all_outputs.extend(outputs)
            all_labels.extend(labels)
            all_outputs_decoded.extend(outputs_decoded)

    # create csv file on predicted vs labels
    df = pd.DataFrame(
        {
            "labels": all_labels,
            "predicted": all_outputs,
            "predicted_decoded": all_outputs_decoded,
        }
    )
    df.to_csv(os.path.join("results", f"{args.run_name}_results.csv"), index=False)



def main():
    args = parse_args()
    random_seed(args.seed)
    ckpt = torch.load(args.pretrained_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        # remove the "module." prefix
        state_dict = {
            k[7:]: v
            for k, v in state_dict.items() if k.startswith("module.")
        }
    else:
        state_dict = ckpt
    
    device_id = init_distributed_device(args)

    tuning_config = ckpt.get("tuning_config").get("tuning_config")
    if tuning_config is None: tuning_config = Config.fromfile(args.tuning_config).get("tuning_config")
    if tuning_config is None:
        print("tuning_config not found in checkpoint")
    else:
        print("tuning_config found in checkpoint: ", tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        model_name="open_flamingo",
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else args.lm_path,
        pretrained_model_path=args.pretrained_path,
        tuning_config=tuning_config,
    )
    model.load_state_dict(state_dict, strict=False)
    # model.half()
    model = model.to("cuda")
    model.eval()

    if args.dataset_config is not None:
        dataset_config = Config.fromfile(args.dataset_config)
    else:
        raise ValueError("dataset_config is required")

    dataset = build_dataset(
        dataset_config=dataset_config.visual_datasets,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )
    test_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=DistributedSampler(dataset, shuffle=False, drop_last=True),
        collate_fn=dataset.collater,
    )

    if dataset_config.get('language_datasets') is not None and len(dataset_config.language_datasets) > 0:
        lang_dataset = build_dataset(
            dataset_config=dataset_config.language_datasets,
            tokenizer=tokenizer,
        )
        lang_dataloader = DataLoader(
            lang_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=InfiniteSampler(lang_dataset, shuffle=True),
            collate_fn=lang_dataset.collater,
        )
        lang_dataloader = iter(lang_dataloader)
    else:
        lang_dataloader = None    

    random_seed(args.seed, args.rank)

    device_id = args.rank % torch.cuda.device_count()
    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    ddp_model.eval()
    eval_model(args, ddp_model, test_dataloader, lang_dataloader, device_id)

if __name__ == "__main__":
    main()