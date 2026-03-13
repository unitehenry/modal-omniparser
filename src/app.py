import modal

app = modal.App("omniparser")

image = (
    modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .apt_install("wget")
        .apt_install("curl")
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
)

@app.function(gpu="h100", image=image)
def model_to_cuda():
    print('model_to_cuda')
