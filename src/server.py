import argparse
import base64
import io
import logging
from typing import List, Optional, Literal, Dict, Union, Tuple

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from nudenet import NudeDetector
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from uvicorn.config import LOGGING_CONFIG

from processing import process_image, save_image_to_base64, LABELS

NudeNetCensor = FastAPI()
detector = NudeDetector()

logger = logging.getLogger("uvicorn")


def is_url(data_str: str) -> bool:
    """检查字符串是否为URL"""
    return data_str.startswith(('http://', 'https://'))


def decode_image_from_base64(img_data: str) -> Image.Image:
    """将Base64编码的图片数据解码为PIL图片"""
    img_data = base64.b64decode(img_data)
    return Image.open(io.BytesIO(img_data))


def load_image_from_url(url: str, proxy: dict = None) -> Image.Image:
    """从URL加载图片并返回PIL图片"""
    response = requests.get(url, proxies=proxy)
    return Image.open(io.BytesIO(response.content))


class ImageConfig(BaseModel):
    """图片处理配置类"""
    proxy: Union[str, Dict[str, str], None] = Field(
        None,
        description="网络代理，可以是字符串或包含http和https键的字典",
        examples=[{"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}]
    )
    mask_type: Literal['color_block', 'gaussian_blur', 'mosaic', 'full_color_block', 'full_gaussian_blur', 'None'] = Field(
        'None',
        description="遮罩类型，可选：'color_block', 'gaussian_blur', 'mosaic', 'full_color_block', 'full_gaussian_blur', 'None'",
        examples=["color_block"]
    )
    mask_color: Tuple[(int, int, int)] = Field(
        (0, 0, 0),
        description="色块颜色（BGR格式）",
        examples=[([0, 0, 0])]
    )
    blur_strength: int = Field(
        4,
        description="模糊强度，建议范围1-10",
        gt=0, le=10
    )
    mask_shape: Literal['rectangle', 'ellipse'] = Field(
        'ellipse',
        description="遮罩形状，可选：'rectangle', 'ellipse'",
        examples=["ellipse"]
    )
    mask_scale: Union[float, int] = Field(
        1.3,
        description="遮罩缩放因子，浮点数",
        gt=0.0
    )
    gradual_ratio: Union[float, int] = Field(
        0.2,
        description="边缘羽化比例，0-1之间的浮点数",
        ge=0.0, le=1.0
    )
    labels: List[str] = Field(
        ["FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"],
        description=f"需要处理的标签列表，可选值：\n{LABELS}"
    )
    score: Union[float, int] = Field(
        0.4,
        description="检测的最小置信度，0-1之间的浮点数",
        ge=0.0, le=1.0
    )


class ImageRequest(BaseModel):
    """图片请求类"""
    image: str = Field(
        ...,
        description="Base64编码的图片字符串或图片URL",
        examples=["http://example.com/image.jpg"],
    )
    config: Optional[ImageConfig] = Field(
        ...,
        description="图片处理配置"
    )


class Detections(BaseModel):
    """检测结果类"""
    __class__: str
    score: float
    box: tuple[int, int, int, int]


class CONTENT(BaseModel):
    """响应内容类"""
    image: Union[str, None]
    detections: list[Detections]


@NudeNetCensor.post("/detect", response_model=CONTENT)
def detect(
        request: ImageRequest,
):
    """处理图片检测请求"""
    image = request.image
    config = request.config
    mask_type = config.mask_type
    proxy = config.proxy

    if image:
        try:
            img = load_image_from_url(image, proxy) if is_url(image) else decode_image_from_base64(image)
            logger.debug("图片加载成功")
        except (IOError, ValueError, TypeError) as e:
            logger.error(f"图片加载错误: {e}")
            raise HTTPException(status_code=400, detail="无效的图片格式")
    else:
        raise HTTPException(status_code=400, detail="未提供图片")

    logger.debug("开始检测过程")
    try:
        img_array = np.array(img)
        detections = detector.detect(img_array)
        logger.debug(f"检测结果: {detections}")

        if detections:
            # 处理图片
            result_img, filtered_detections = process_image(img, detections, logger, config)
            
            if filtered_detections:
                logger.info(f"过滤后的检测结果: {filtered_detections}")
                encoded_img = save_image_to_base64(result_img)
                
                if mask_type == 'None':
                    content = {'detections': filtered_detections}
                else:
                    content = {'image': encoded_img, 'detections': filtered_detections}
            else:
                # 如果没有超过阈值的检测结果，返回原始图片
                encoded_img = save_image_to_base64(img)
                content = {'image': encoded_img, 'detections': []}
            
            return content
        return JSONResponse(content={'warn': '未检测到内容'}, status_code=202)
    except Exception as e:
        logger.error(f"处理过程中出错: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="图片处理错误")


@NudeNetCensor.post("/detect_file")
async def detect_file(
        file: UploadFile = File(...),
):
    """处理文件上传检测请求"""
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except (IOError, ValueError, TypeError):
        raise HTTPException(status_code=400, detail="无效的文件格式")

    img_array = np.array(img)
    detections = detector.detect(img_array)

    if detections:
        result_img, _ = process_image(img, detections, logger)
        return StreamingResponse(
            io.BytesIO(save_image_to_base64(result_img).encode()),
            media_type="image/png"
        )
    else:
        return JSONResponse(content={'warn': '未检测到内容'}, status_code=202)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="NudeNetCensor服务器")
    parser.add_argument("--log-level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="设置日志级别")
    parser.add_argument("--port", type=int, default=15000, help="设置服务器端口")

    args = parser.parse_args()
    log_level = args.log_level

    logger.setLevel(log_level)

    current_level = logger.getEffectiveLevel()
    print(f"当前日志级别: {logging.getLevelName(current_level)}")

    import uvicorn

    # Uvicorn日志配置
    LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = log_level

    uvicorn.run("server:NudeNetCensor", host="0.0.0.0", port=args.port, reload=False, log_config=LOGGING_CONFIG,
                log_level=args.log_level.lower())
