import logging
import io
import base64
from typing import List, Dict, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# 可检测的标签列表
LABELS = [
    "FEMALE_GENITALIA_COVERED", "FACE_FEMALE", "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED", "FEET_EXPOSED", "BELLY_COVERED", "FEET_COVERED",
    "ARMPITS_COVERED", "ARMPITS_EXPOSED", "FACE_MALE", "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED", "ANUS_COVERED", "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED"
]

# 检测框颜色列表（BGR格式）
COLORS_BGR = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 199)
]

# 标签和颜色的映射关系
LABEL_COLOR_MAP = {label: color for label, color in zip(LABELS, COLORS_BGR)}

log = logging.getLogger("uvicorn")


def apply_feathering(alpha_channel, mask_shape, gradient_ratio):
    """应用边缘羽化效果"""
    h, w = alpha_channel.shape
    for i in range(h):
        for j in range(w):
            if mask_shape == 'ellipse':
                # 计算到椭圆中心的距离
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
                # 计算到矩形边缘的距离
                dist_x = min(j, w - j - 1) / (w / 2)
                dist_y = min(i, h - i - 1) / (h / 2)
                dist = 1 - min(dist_x, dist_y)
                if dist + gradient_ratio <= 1:
                    alpha = 255
                else:
                    alpha = int(255 * (1 - (dist + gradient_ratio - 1) / gradient_ratio))
            else:
                alpha = 255  # 默认不透明

            alpha_channel[i, j] = alpha
    return alpha_channel


def draw_detections(img, detections):
    """在图片上绘制检测框和标签"""
    log.debug('开始绘制检测框...')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 15) if ImageFont.truetype("arial.ttf", 15) else ImageFont.load_default()

    for detection in detections:
        box = detection["box"]
        label = detection["class"]
        score = detection["score"]
        color = LABEL_COLOR_MAP.get(label, (255, 255, 255))  # 默认白色
        x, y, w, h = box

        # 绘制检测框
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

        # 创建带置信度的标签文本
        label_text = f"{label} ({score:.3f})"

        # 绘制标签背景和文本
        text_bbox = draw.textbbox((x, y - 10), label_text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([x, y - text_height - 10, x + text_width, y - 10], fill=color)
        draw.text((x, y - text_height - 10), label_text, fill=(255, 255, 255), font=font)

    return img


def apply_color_block(image, box, color, mask_shape, mask_scale, gradient_ratio):
    """应用色块遮罩"""
    log.debug('开始应用色块遮罩...')
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

    # 创建透明度通道
    alpha_channel = np.array(mask)

    # 应用羽化效果
    alpha_channel = apply_feathering(alpha_channel, mask_shape, gradient_ratio)

    # 将透明度通道转换回图片并用作遮罩
    mask = Image.fromarray(alpha_channel)

    image.paste(color_image, (x, y), mask=mask)
    return image


def apply_gaussian_blur(image, box, blur_strength, mask_shape, mask_scale, gradient_ratio):
    """应用高斯模糊遮罩"""
    log.debug('开始应用高斯模糊遮罩...')
    x, y, w, h = [int(coord) for coord in [box[0] - box[2] * (mask_scale - 1) / 2,
                                           box[1] - box[3] * (mask_scale - 1) / 2,
                                           box[2] * mask_scale,
                                           box[3] * mask_scale]]

    roi = image.crop((x, y, x + w, y + h)).convert("RGBA")
    blurred_roi = roi.filter(ImageFilter.GaussianBlur(blur_strength))

    roi_array = np.array(blurred_roi)

    # 创建透明度通道
    alpha_channel = np.zeros((h, w), dtype=np.uint8)
    alpha_channel = apply_feathering(alpha_channel, mask_shape, gradient_ratio)

    # 将透明度通道应用到ROI数组
    roi_array[..., 3] = alpha_channel

    feathered_roi = Image.fromarray(roi_array)
    image.paste(feathered_roi, (x, y), feathered_roi)
    return image


def apply_mosaic(image, box, blur_strength, mask_shape, mask_scale):
    """应用马赛克遮罩"""
    log.debug('开始应用马赛克遮罩...')
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

    # 应用马赛克效果
    for i in range(0, mosaic_array.shape[0], mosaic_size):
        for j in range(0, mosaic_array.shape[1], mosaic_size):
            mosaic_array[i:i + mosaic_size, j:j + mosaic_size] = np.mean(
                mosaic_array[i:i + mosaic_size, j:j + mosaic_size], axis=(0, 1), dtype=int)

    mosaic_image = Image.fromarray(mosaic_array)
    image.paste(mosaic_image, (x, y), mask)
    return image


def apply_full_color_block(image, color):
    """应用全图色块遮罩"""
    log.debug('开始应用全图色块遮罩...')
    color_block = Image.new('RGB', image.size, tuple(int(c) for c in color))
    mask = Image.new('L', image.size, 255)
    image.paste(color_block, (0, 0), mask)
    return image


def apply_full_gaussian_blur(image, blur_strength):
    """应用全图高斯模糊遮罩"""
    log.debug('开始应用全图高斯模糊遮罩...')
    return image.filter(ImageFilter.GaussianBlur(blur_strength))


def save_image_to_base64(image: Image.Image) -> str:
    """将PIL图片保存为Base64编码"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def filter_detections(detections: List[Dict], config) -> List[Dict]:
    """过滤检测结果，只保留超过阈值的检测"""
    filtered_detections = []
    for detection in detections:
        if detection['class'] in config.labels and detection['score'] >= config.score:
            filtered_detections.append(detection)
    return filtered_detections


def process_image(image: Image.Image, detections: List[Dict], logger, config=None) -> Tuple[Image.Image, List[Dict]]:
    """处理图片，根据配置应用相应的遮罩效果"""
    global log
    log = logger

    if not config:
        image = draw_detections(image, detections)
        return image, detections

    # 过滤检测结果
    filtered_detections = filter_detections(detections, config)

    if not filtered_detections:
        return image, []

    # 获取配置参数
    mask_type = config.mask_type
    mask_color = config.mask_color
    blur_strength = config.blur_strength * 10
    mask_shape = config.mask_shape
    mask_scale = config.mask_scale
    gradual_ratio = config.gradual_ratio

    # 处理每个检测结果
    for detection in filtered_detections:
        box = detection['box']

        if mask_type == 'color_block':
            image = apply_color_block(image, box, mask_color, mask_shape, mask_scale, gradual_ratio)
        elif mask_type == 'full_color_block':
            image = apply_full_color_block(image, mask_color)
        elif mask_type == 'gaussian_blur':
            image = apply_gaussian_blur(image, box, blur_strength, mask_shape, mask_scale, gradual_ratio)
        elif mask_type == 'full_gaussian_blur':
            image = apply_full_gaussian_blur(image, blur_strength)
        elif mask_type == 'mosaic':
            image = apply_mosaic(image, box, blur_strength, mask_shape, mask_scale)

    return image, filtered_detections
