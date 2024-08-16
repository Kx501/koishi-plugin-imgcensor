import base64
import io
from typing import List, Optional, Literal, Dict, Union, Tuple

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from nudenet import NudeDetector
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from processing import process_image, LABELS

NudeNetCensor = FastAPI()
detector = NudeDetector()


# Helper function to check if the string is a URL
def is_url(data_str: str) -> bool:
    return data_str.startswith(('http://', 'https://'))


# Decode Base64 image data to PIL image
def decode_image_from_base64(img_data: str) -> Image.Image:
    img_data = base64.b64decode(img_data)
    return Image.open(io.BytesIO(img_data))


# Load image from URL and return PIL image
def load_image_from_url(url: str, proxy: dict = None) -> Image.Image:
    response = requests.get(url, proxies=proxy)
    return Image.open(io.BytesIO(response.content))


class ImageConfig(BaseModel):
    proxy: Union[str, Dict[str, str], None] = Field(
        None,
        description="Network proxy, can be a string or a dict with http and https keys",
        examples=[{"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}]
    )
    mask_type: Literal['color_block', 'gaussian_blur', 'mosaic', 'full_color_block', 'full_gaussian_blur', 'None'] = Field(
        'None',
        description="Mask shape, one of 'color_block', 'gaussian_blur', 'mosaic', 'full_color_block', 'full_gaussian_blur', 'None'",
        examples=["color_block"]
    )
    mask_color: Optional[Tuple[(int, int, int)]] = Field(
        (0, 0, 0),
        description="Color block color in BGR format",
        examples=[([0, 0, 0])]
    )
    blur_strength: Optional[int] = Field(
        4,
        description="Blur strength, recommended range is 1 to 10",
        gt=0, le=10
    )
    mask_shape: Literal['rectangle', 'ellipse'] = Field(
        'ellipse',
        description="Mask shape, one of 'rectangle', 'ellipse'",
        examples=["ellipse"]
    )
    mask_scale: Union[float, int] = Field(
        1.3,
        description="Mask scale factor, a floating point number",
        gt=0.0
    )
    gradual_ratio: Optional[Union[float, int]] = Field(
        0.2,
        description="Edge feathering ratio, a floating point number between 0 and 1",
        ge=0.0, le=1.0
    )
    labels: Optional[List[str]] = Field(
        ["FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"],
        description=f"Labels that need to be processed, optional list:\n{LABELS}"
    )


class ImageRequest(BaseModel):
    image: Optional[str] = Field(
        None,
        description="Base64 encoded string or URL of the image",
        examples=["http://example.com/image.jpg"],
    )
    config: Optional[ImageConfig] = Field(
        ...,
        description="Configuration for the image processing"
    )


@NudeNetCensor.post("/detect")
async def detect(
        request: ImageRequest,
):
    image = request.image
    config = request.config
    proxy = config.proxy

    if image:
        try:
            img = load_image_from_url(image, proxy) if is_url(image) else decode_image_from_base64(image)
        except (IOError, ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid image format")
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    # Convert PIL image to numpy array for detection
    img_array = np.array(img)
    detections = detector.detect(img_array)

    # Process image if detections are found
    if detections:
        result_img, filtered_detections = process_image(img, detections, config)

        if filtered_detections and config.mask_type != 'None':
            buffer = io.BytesIO()
            result_img.save(buffer, format='PNG')
            encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return JSONResponse(content={'detections': filtered_detections, 'images': [encoded_img]})
        elif config.mask_type == 'None':
            return JSONResponse(content={'detections': filtered_detections})
        else:
            return JSONResponse(content={'warn': 'No selected tags detected'}, status_code=204)
    else:
        return JSONResponse(content={'warn': 'No detected'}, status_code=204)


@NudeNetCensor.post("/detect_file")
async def detect_file(
        file: UploadFile = File(...),
):
    # Determine input type and load the image
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except (IOError, ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid file format")

    # Convert PIL image to numpy array for detection
    img_array = np.array(img)
    detections = detector.detect(img_array)

    # Process image if detections are found
    if detections:
        result_img = process_image(img, detections)
        buffer = io.BytesIO()
        result_img.save(buffer, format='PNG')
        buffer.seek(0)
        # 返回文件流
        return StreamingResponse(buffer, media_type="image/png")
    else:
        return JSONResponse(content={'warn': 'No detected'}, status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:NudeNetCensor", host="0.0.0.0", port=5000, reload=False)
