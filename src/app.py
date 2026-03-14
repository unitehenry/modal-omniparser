import functools
import modal
from functools import wraps

app = modal.App("omniparser")
vol = modal.Volume.from_name("omniparser", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["wget", "curl", "libgl1", "libglib2.0-0"])
    .shell(["/bin/bash", "-c"])
    .env({"CONDA_DIR": "/opt/conda"})
    .run_commands(
        [
            "mkdir -p $CONDA_DIR",
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $CONDA_DIR/miniconda.sh",
            "bash $CONDA_DIR/miniconda.sh -b -u -p $CONDA_DIR",
            "rm $CONDA_DIR/miniconda.sh",
            "$CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
            "$CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
            "$CONDA_DIR/bin/conda init --all",
        ]
    )
    .run_commands(
        [
            "wget https://github.com/microsoft/OmniParser/archive/refs/tags/v.2.0.1.tar.gz -O OmniParser.tar.gz",
            "tar -xzf OmniParser.tar.gz",
            "mv OmniParser-* OmniParser",
        ]
    )
    .workdir("OmniParser")
    .run_commands(
        [
            "sed -i 's/\<paddleocr\>/paddleocr<=2.8/g' requirements.txt",
            "sed -i 's/\<torch\>/torch==2.2.0/g' requirements.txt",
            "sed -i 's/\<transformers\>/transformers==4.38.2/g' requirements.txt",
            "sed -i 's/\<paddlepaddle\>/paddlepaddle<=2.8/g' requirements.txt",
            "$CONDA_DIR/bin/conda create -n omni python==3.12 -y",
            "$CONDA_DIR/bin/conda run -n omni pip install -r requirements.txt",
            "$CONDA_DIR/bin/conda install -n omni cuda -c nvidia/label/cuda-12.2.0 -y",
        ]
    )
    .env(
        {"PATH": "$CONDA_DIR/envs/omni/bin:$PATH", "CUDA_HOME": "$CONDA_DIR/envs/omni"}
    )
    .run_commands(
        [
            "wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.0/flash_attn-2.6.0+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
            "$CONDA_DIR/bin/conda run -n omni pip install --no-dependencies flash_attn-2.6.0+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
        ]
    )
    .run_commands("curl -LsSf https://hf.co/cli/install.sh | bash")
    .run_commands(
        [
            "; ".join(
                [
                    "for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}",
                    'do env PATH="/root/.local/bin:$PATH" hf download microsoft/OmniParser-v2.0 "$f" --local-dir weights',
                    "done",
                ]
            ),
            "mv weights/icon_caption weights/icon_caption_florence",
        ]
    )
)


def cache(func):
    import os
    import subprocess

    @wraps(func)
    def wrapper(*args, **kwargs):
        subprocess.run(["mv", "-f", "--", "/data/.paddleocr", "/root"], check=False)

        subprocess.run(["mv", "-f", "--", "/data/.EasyOCR", "/root"], check=False)

        subprocess.run(
            [
                "mv",
                "-f",
                "--",
                "/data/.config/Ultralytics",
                "/root/.config/Ultralytics",
            ],
            check=False,
        )

        subprocess.run(
            [
                "mv",
                "-f",
                "--",
                "/data/.cache/huggingface/hub",
                "/root/.cache/huggingface/hub",
            ],
            check=False,
        )

        try:
            return func(*args, **kwargs)
        finally:
            subprocess.run(
                ["mv", "-f", "--", "/root/.paddleocr", "/data/.paddleocr"], check=False
            )

            subprocess.run(
                ["mv", "-f", "--", "/root/.EasyOCR", "/data/.EasyOCR"], check=False
            )

            os.makedirs(os.path.dirname("/data/.config/Ultralytics"), exist_ok=True)

            subprocess.run(
                [
                    "mv",
                    "-f",
                    "--",
                    "/root/.config/Ultralytics",
                    "/data/.config/Ultralytics",
                ],
                check=False,
            )

            os.makedirs(os.path.dirname("/data/.cache/huggingface/hub"), exist_ok=True)

            subprocess.run(
                [
                    "mv",
                    "-f",
                    "--",
                    "/root/.cache/huggingface/hub",
                    "/data/.cache/huggingface/hub",
                ],
                check=False,
            )

    return wrapper


@app.function(gpu="h100", image=image, volumes={"/data": vol})
@cache
def parse():
    import time
    import importlib
    from util.utils import (
        get_som_labeled_img,
        check_ocr_box,
        get_caption_model_processor,
        get_yolo_model,
    )
    import torch
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

    image_path = "imgs/google_page.png"
    image_path = "imgs/windows_home.png"
    # image_path = 'imgs/windows_multitab.png'
    # image_path = 'imgs/omni3.jpg'
    # image_path = 'imgs/ios.png'
    image_path = "imgs/word.png"
    # image_path = 'imgs/excel2.png'

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

    return parsed_content_list

if __name__ == '__main__':
    with app.run():
        print(parse.remote())
