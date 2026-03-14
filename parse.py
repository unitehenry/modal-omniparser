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

    # image_path = "imgs/google_page.png"
    # image_path = "imgs/windows_home.png"
    # image_path = 'imgs/windows_multitab.png'
    # image_path = 'imgs/omni3.jpg'
    # image_path = 'imgs/ios.png'
    # image_path = 'imgs/excel2.png'
    image_path = "imgs/word.png"

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
