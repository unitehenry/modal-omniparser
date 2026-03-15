import modal
from functools import wraps

omniparser = modal.Volume.from_name("omniparser", create_if_missing=True)


def cache(func):
    import os
    import subprocess

    CACHE_PATHS = [
        ".paddleocr",
        ".EasyOCR",
        ".config",
        ".cache",
    ]

    @wraps(func)
    def wrapper(*args, **kwargs):
        for path in CACHE_PATHS:
            subprocess.run(["cp", "-rf", "--", f"/data/{path}", "/root"], check=False)

        try:
            return func(*args, **kwargs)
        finally:
            for path in CACHE_PATHS:
                os.makedirs(os.path.dirname(f"/data/{path}"), exist_ok=True)

                subprocess.run(
                    ["cp", "-rf", "--", f"/root/{path}", f"/data/{path}"], check=False
                )

            omniparser.commit()

    return wrapper
