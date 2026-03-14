import modal

omniparser_v2_0_1 = (
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
    .add_local_file("omniparser-requirements.txt", "/etc/omniparser-requirements.txt", copy=True)
    .run_commands(
        [
            "cp /etc/omniparser-requirements.txt requirements.txt",
            "cat requirements.txt",
            "$CONDA_DIR/bin/conda create -n omni python==3.12 -y",
            "$CONDA_DIR/bin/conda run -n omni pip install -r requirements.txt",
            "$CONDA_DIR/bin/conda install -n omni cuda -c nvidia/label/cuda-12.2.0 -y",
        ])
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
    .add_local_python_source("volume")
    .add_local_python_source("image")
    .add_local_python_source("parse")
    .add_local_python_source("app")
)
