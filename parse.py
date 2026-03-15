import app
import image
import volume


@app.app.function(
    gpu="h100", image=image.omniparser_v2_0_1, volumes={"/data": volume.vol}
)
@volume.cache
def parse(file_url: str):
    import time
    import json
    import importlib
    import urllib.request
    import tempfile
    import os
    import io
    import base64
    import subprocess
    from util.utils import (
        get_som_labeled_img,
        check_ocr_box,
        get_caption_model_processor,
        get_yolo_model,
    )
    import torch
    from urllib.parse import urlparse
    from ultralytics import YOLO
    from PIL import Image

    device = "cuda"

    model_path = "weights/icon_detect/model.pt"

    som_model = get_yolo_model(model_path)

    som_model.to(device)

    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence",
        device=device,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_path = tmp_file.name

    urllib.request.urlretrieve(file_url, tmp_path)

    image_path = tmp_path

    image = Image.open(image_path)

    image_rgb = image.convert("RGB")

    box_overlay_ratio = max(image.size) / 3200

    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    BOX_TRESHOLD = 0.05

    start = time.time()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=True,
    )

    text, ocr_bbox = ocr_bbox_rslt

    cur_time_ocr = time.time()

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path,
        som_model,
        BOX_TRESHOLD=BOX_TRESHOLD,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=0.7,
        scale_img=False,
        batch_size=128,
    )

    cur_time_caption = time.time()

    labeled_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))

    os.makedirs("/root/output", exist_ok=True)

    output_file_name = f"{os.path.basename(urlparse(file_url).path).split('.')[0]}.png"

    output_file_path = f"/root/output/{output_file_name}"

    with open(output_file_path, "wb") as output_file:
        labeled_image.save(output_file.name)

    os.makedirs("/data/output", exist_ok=True)

    subprocess.run(
        ["mv", "-f", "--", output_file_path, f"/data/output/{output_file_name}"],
        check=False,
    )

    return json.dumps(parsed_content_list)
