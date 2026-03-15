import modal
from functools import wraps

vol = modal.Volume.from_name("omniparser", create_if_missing=True)

def save_dir(dir_name : str):
    import os
    import subprocess

    print(dir_name)

    print(["mv", "-f", "--", dir_name, f"/data/output"])

    print(subprocess.run(
        ["mv", "-f", "--", dir_name, f"/data/output"], check=False, capture_output=True, text=True
    ))

    vol.commit()

def cache(func):
    import os
    import subprocess

    CACHE_PATHS = [
        ".paddleocr",
        ".EasyOCR",
        ".config/Ultralytics",
        ".cache/huggingface/hub",
    ]

    @wraps(func)
    def wrapper(*args, **kwargs):
        for path in CACHE_PATHS:
            subprocess.run(["mv", "-f", "--", f"/data/{path}", "/root"], check=False)

        try:
            return func(*args, **kwargs)
        finally:
            for path in CACHE_PATHS:
                os.makedirs(os.path.dirname(f"/data/{path}"), exist_ok=True)

                subprocess.run(
                    ["mv", "-f", "--", f"/root/{path}", f"/data/{path}"], check=False
                )

            vol.commit()

    return wrapper
