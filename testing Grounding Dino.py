import argparse, os, sys, argparse, json, torch, cv2, torchvision

from typing import Any, Dict, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

def load_image(image_path): #loads image from path
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image in PIL

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None) #transform to a known size to speed up processing and normalise
    return image_pil, image #returns transformed and non transformed image


def load_model(model_config_path, model_checkpoint_path): #loads existing model
    args = SLConfig.fromfile(model_config_path)
    args.device = "cpu" #sets device to CPU (can use "CUDA")
    model = build_model(args) #builds model for CPU
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu") #maps to CPU
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval() #allows evaluation to run but dumps output
    return model #returns model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=True, token_spans=None): #I have no idea what the section does.
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cpu" #sets device to CPU (Can use "CUDA", not working in my setup)
    model = model.to(device) #Loads model to CPU
    image = image.to(device) #Loads image to CPU
    with torch.no_grad(): #running with torch with no gradient
        outputs = model(image[None], captions=[caption]) #outputs image and caption
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4) outputs prediction boxes

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases
    return boxes_filt, pred_phrases


def plot_boxes(image_pil, tgt): #creates bounding boxes around targets
    H, W = tgt["size"] #size of image
    boxes = tgt["boxes"] #prediction boxes
    labels = tgt["labels"] #box lables
    assert len(boxes) == len(labels), "boxes and labels must have same length" #ensures every box has a label 
    #print(boxes) #this is some sort of insane ratio of positions maybe in y, hig, x, wid? the output is nonsence

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        intbox = [x0, y0, x1, y1]
        print(intbox)
        return box

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--image_path", "-i", type=str, required=True, help="path to image file")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    args = parser.parse_args()

    # cfg
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "weights/groundingdino_swint_ogc.pth"
    image_path = args.image_path
    text_prompt = "Person"
    output_dir = args.output_dir
    box_threshold = 0.3
    text_threshold = 0.25

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, checkpoint_path)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))


    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    # import ipdb; ipdb.set_trace()
    boxes = plot_boxes(image_pil, pred_dict)