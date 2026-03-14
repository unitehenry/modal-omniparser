import modal

app = modal.App("omniparser")
vol = modal.Volume.from_name("omniparser", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "wget", "curl", "libgl1", "libglib2.0-0"])
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
    .run_commands("git clone https://github.com/microsoft/OmniParser")
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
            # "$CONDA_DIR/bin/conda run -n omni pip install flash-attn==2.5.8 --no-build-isolation"
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


def setup():
    import subprocess

    subprocess.run(["mv", "-f", "--", "/data/.paddleocr", "/root"], check=False)
    subprocess.run(["mv", "-f", "--", "/data/.EasyOCR", "/root"], check=False)


def cleanup():
    import subprocess

    subprocess.run(
        ["mv", "-f", "--", "/root/.paddleocr", "/data/.paddleocr"], check=False
    )
    subprocess.run(["mv", "-f", "--", "/root/.EasyOCR", "/data/.EasyOCR"], check=False)


@app.function(gpu="h100", image=image, volumes={"/data": vol})
def model_to_cuda():
    try:
        setup()

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

        print("model to {}".format(device))

        get_caption_model_processor(
            model_name="florence2",
            model_name_or_path="weights/icon_caption_florence",
            device="cuda",
        )
    finally:
        cleanup()
