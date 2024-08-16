from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
]

COLORS_BGR = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 199)
]

LABEL_COLOR_MAP = {label: color for label, color in zip(LABELS, COLORS_BGR)}


def apply_feathering(alpha_channel, mask_shape, gradient_ratio):
    h, w = alpha_channel.shape
    for i in range(h):
        for j in range(w):
            if mask_shape == 'ellipse':
                dist_x = ((j - w / 2) / (w / 2)) ** 2
                dist_y = ((i - h / 2) / (h / 2)) ** 2
                dist = np.sqrt(dist_x + dist_y)
                if dist > 1:
                    alpha = 0
                elif dist + gradient_ratio <= 1:
                    alpha = 255
                else:
                    alpha = int(255 * (1 - (dist + gradient_ratio - 1) / gradient_ratio))
            elif mask_shape == 'rectangle':
                dist_x = min(j, w - j - 1) / (w / 2)
                dist_y = min(i, h - i - 1) / (h / 2)
                dist = 1 - min(dist_x, dist_y)
                if dist + gradient_ratio <= 1:
                    alpha = 255
                else:
                    alpha = int(255 * (1 - (dist + gradient_ratio - 1) / gradient_ratio))
            else:
                alpha = 255  # Default case if mask_shape is not recognized

            alpha_channel[i, j] = alpha
    return alpha_channel


def draw_detections(img, detections):
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 15) if ImageFont.truetype("arial.ttf", 15) else ImageFont.load_default()

    for detection in detections:
        box = detection["box"]
        label = detection["class"]
        score = detection["score"]
        color = LABEL_COLOR_MAP.get(label, (255, 255, 255))  # Default color is white
        x, y, w, h = box

        # Draw detection box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # Create label text with confidence score
        label_text = f"{label} ({score:.3f})"

        # Draw label background and text
        text_bbox = draw.textbbox((x, y - 10), label_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - text_height - 10, x + text_width, y - 10], fill=color)
        draw.text((x, y - text_height - 10), label_text, fill=(255, 255, 255), font=font)

    # Save the processed image
    # img.save('detection_result.png')
    return img


def apply_color_block(image, box, color, mask_shape, mask_scale, gradient_ratio):
    x, y, w, h = [int(coord) for coord in [box[0] - box[2] * (mask_scale - 1) / 2,
                                           box[1] - box[3] * (mask_scale - 1) / 2,
                                           box[2] * mask_scale,
                                           box[3] * mask_scale]]

    color_image = Image.new('RGB', (w, h), tuple(int(c) for c in color))
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    if mask_shape == 'rectangle':
        draw.rectangle([0, 0, w, h], fill=255)
    elif mask_shape == 'ellipse':
        draw.ellipse([0, 0, w, h], fill=255)

    # Create an alpha channel to store transparency changes
    alpha_channel = np.array(mask)

    # Apply feathering
    alpha_channel = apply_feathering(alpha_channel, mask_shape, gradient_ratio)

    # Convert alpha_channel back to Image and use it as the mask
    mask = Image.fromarray(alpha_channel)

    image.paste(color_image, (x, y), mask=mask)
    return image


def apply_gaussian_blur(image, box, blur_strength, mask_shape, mask_scale, gradient_ratio):
    x, y, w, h = [int(coord) for coord in [box[0] - box[2] * (mask_scale - 1) / 2,
                                           box[1] - box[3] * (mask_scale - 1) / 2,
                                           box[2] * mask_scale,
                                           box[3] * mask_scale]]

    roi = image.crop((x, y, x + w, y + h)).convert("RGBA")
    blurred_roi = roi.filter(ImageFilter.GaussianBlur(blur_strength))

    roi_array = np.array(blurred_roi)

    # Create an alpha channel to store transparency changes
    alpha_channel = np.zeros((h, w), dtype=np.uint8)
    alpha_channel = apply_feathering(alpha_channel, mask_shape, gradient_ratio)

    # Apply alpha channel to roi_array
    roi_array[..., 3] = alpha_channel

    feathered_roi = Image.fromarray(roi_array)
    image.paste(feathered_roi, (x, y), feathered_roi)
    return image


def apply_mosaic(image, box, blur_strength, mask_shape, mask_scale):
    x, y, w, h = [int(coord) for coord in [box[0] - box[2] * (mask_scale - 1) / 2,
                                           box[1] - box[3] * (mask_scale - 1) / 2,
                                           box[2] * mask_scale,
                                           box[3] * mask_scale]]

    mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)

    if mask_shape == 'rectangle':
        mask_draw.rectangle([0, 0, w, h], fill=255)
    elif mask_shape == 'ellipse':
        mask_draw.ellipse([0, 0, w, h], fill=255)

    roi = image.crop((x, y, x + w, y + h))
    mosaic_array = np.array(roi)
    mosaic_size = max(1, blur_strength)

    for i in range(0, mosaic_array.shape[0], mosaic_size):
        for j in range(0, mosaic_array.shape[1], mosaic_size):
            mosaic_array[i:i + mosaic_size, j:j + mosaic_size] = np.mean(
                mosaic_array[i:i + mosaic_size, j:j + mosaic_size], axis=(0, 1), dtype=int)

    mosaic_image = Image.fromarray(mosaic_array)
    image.paste(mosaic_image, (x, y), mask)
    return image


def apply_full_color_block(image, color):
    color_block = Image.new('RGB', image.size, tuple(int(c) for c in color))
    mask = Image.new('L', image.size, 255)
    image.paste(color_block, (0, 0), mask)
    return image


def apply_full_gaussian_blur(image, blur_strength):
    return image.filter(ImageFilter.GaussianBlur(blur_strength))


def process_image(image, detections, config=None):
    # Save original picture
    # image.save('original_image.png')

    if config:
        mask_type = config.mask_type
        mask_color = config.mask_color
        blur_strength = config.blur_strength * 10
        mask_shape = config.mask_shape
        mask_scale = config.mask_scale
        gradual_ratio = config.gradual_ratio
        target_labels = config.labels
    else:
        image = draw_detections(image, detections)
        return image

    filtered_detections = []
    for detection in detections:
        # print(detection)

        if detection['class'] not in target_labels:
            continue

        print(detection)

        filtered_detections.append(detection)
        box = detection['box']

        if mask_type == 'color_block':
            image = apply_color_block(image, box, mask_color, mask_shape, mask_scale, gradual_ratio)
        elif mask_type == 'full_color_block':
            image = apply_full_color_block(image, mask_color)
        elif mask_type == 'gaussian_blur':
            image = apply_gaussian_blur(image, box, blur_strength, mask_shape, mask_scale, gradual_ratio)
        elif mask_type == 'full_gaussian_blur':
            # 1 * 10
            image = apply_full_gaussian_blur(image, blur_strength)
        elif mask_type == 'mosaic':
            # 1 * 10
            image = apply_mosaic(image, box, blur_strength, mask_shape, mask_scale)

    # Save the processed picture
    image.save('detection_result.png')
    return image, filtered_detections
