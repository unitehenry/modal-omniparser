import modal

app = modal.App("omniparser")

image = (
    modal.Image.debian_slim(python_version="3.12")
        .apt_install([ "git", "wget", "curl", "libgl1", "libglib2.0-0" ])
        .shell(["/bin/bash", "-c"])
        .env({ 'CONDA_DIR': '/opt/conda' })
        .run_commands([
            "mkdir -p $CONDA_DIR",
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $CONDA_DIR/miniconda.sh",
            "bash $CONDA_DIR/miniconda.sh -b -u -p $CONDA_DIR",
            "rm $CONDA_DIR/miniconda.sh",
            "$CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
            "$CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
            "$CONDA_DIR/bin/conda init --all"
        ])
        .run_commands("git clone https://github.com/microsoft/OmniParser")
        .workdir("OmniParser")
        .run_commands([
            "$CONDA_DIR/bin/conda create -n 'omni' python==3.12 -y",
            "$CONDA_DIR/bin/conda run -n omni pip install -r requirements.txt",
        ])
        .run_commands("curl -LsSf https://hf.co/cli/install.sh | bash")
        .run_commands([
            'for f in icon_detect/{train_args.yaml,model.pt,model.yaml} icon_caption/{config.json,generation_config.json,model.safetensors}; do env PATH="/root/.local/bin:$PATH" hf download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done',
            "mv weights/icon_caption weights/icon_caption_florence"
        ])
        .env({ "PATH": "$CONDA_DIR/envs/omni/bin:$PATH" })
)

@app.function(gpu="h100", image=image)
def model_to_cuda():
    from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
    import torch
    from ultralytics import YOLO
    from PIL import Image

    device = 'cuda'

    model_path='weights/icon_detect/model.pt'

    som_model = get_yolo_model(model_path)

    som_model.to(device)

    print('model to {}'.format(device))
