import io
import imghdr
from typing import List
import PIL.Image
import ocr_service_pb2 as pb2
import logging

logger = logging.getLogger(__name__)

def validate_image_data(image_data: bytes) -> bool:
    """Validate image data using imghdr."""
    try:
        image_type = imghdr.what(None, h=image_data)
        logger.debug(f"Detected image type: {image_type}")
        return image_type is not None
    except Exception as e:
        logger.error(f"Error validating image data: {str(e)}")
        return False

def load_image(request) -> PIL.Image.Image:
    """Load image from request bytes with enhanced validation and error handling."""
    try:
        # Validate image data
        if not validate_image_data(request.image_data):
            raise ValueError("Invalid image format or corrupted image data")

        # Create BytesIO object and log its initial state
        image_bytes = io.BytesIO(request.image_data)
        logger.debug(f"Created BytesIO object: position={image_bytes.tell()}, size={len(request.image_data)}")
        
        # Force image format detection by seeking to start
        image_bytes.seek(0)
        logger.debug("Seeking to start of BytesIO buffer")
        
        # Try to identify image format
        try:
            image = PIL.Image.open(image_bytes)
            logger.debug(f"PIL.Image.open successful: format={image.format}")
        except Exception as e:
            logger.error(f"Failed to open image with PIL: {str(e)}")
            raise
        
        # Load image data immediately to catch potential format errors
        try:
            image.load()
            logger.debug(f"Image loaded successfully: size={image.size}, mode={image.mode}")
        except Exception as e:
            logger.error(f"Failed to load image data: {str(e)}")
            raise
        
        return image
    except Exception as e:
        error_msg = f"Failed to load image: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise ValueError(error_msg)

def create_detected_element(
    bbox: List[float],
    confidence: float = 1.0,
    reading_order: int = 0
) -> pb2.DetectedElement:
    """Create a DetectedElement from detection results."""
    element = pb2.DetectedElement()

    if not bbox:
        raise ValueError("Bbox cannot be None or empty")

    if isinstance(bbox[0], (list, tuple)):
        bbox = [coord for sublist in bbox for coord in sublist][:4]

    if len(bbox) < 4:
        raise ValueError(f"Invalid bbox format: {bbox}")

    # Ensure all bbox coordinates are valid numbers
    try:
        element.bbox.x1 = float(bbox[0] or 0)
        element.bbox.y1 = float(bbox[1] or 0)
        element.bbox.x2 = float(bbox[2] or 0)
        element.bbox.y2 = float(bbox[3] or 0)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid bbox coordinate values: {bbox}. Error: {str(e)}")

    # Ensure confidence is a valid float
    try:
        element.confidence = float(confidence if confidence is not None else 1.0)
    except (TypeError, ValueError):
        element.confidence = 1.0

    element.reading_order = int(reading_order if reading_order is not None else 0)
    return element

# Map Surya labels to proto enum types remains unchanged
LABEL_TO_TYPE = {
    'Blank': pb2.LayoutElement.ElementType.BLANK,
    'Text': pb2.LayoutElement.ElementType.TEXT,
    'TextInlineMath': pb2.LayoutElement.ElementType.TEXT_INLINE_MATH,
    'Code': pb2.LayoutElement.ElementType.CODE,
    'SectionHeader': pb2.LayoutElement.ElementType.SECTION_HEADER,
    'Caption': pb2.LayoutElement.ElementType.CAPTION,
    'Footnote': pb2.LayoutElement.ElementType.FOOTNOTE,
    'Equation': pb2.LayoutElement.ElementType.EQUATION,
    'ListItem': pb2.LayoutElement.ElementType.LIST_ITEM,
    'PageFooter': pb2.LayoutElement.ElementType.PAGE_FOOTER,
    'PageHeader': pb2.LayoutElement.ElementType.PAGE_HEADER,
    'Picture': pb2.LayoutElement.ElementType.PICTURE,
    'Figure': pb2.LayoutElement.ElementType.FIGURE,
    'Table': pb2.LayoutElement.ElementType.TABLE,
    'Form': pb2.LayoutElement.ElementType.FORM,
    'TableOfContents': pb2.LayoutElement.ElementType.TABLE_OF_CONTENTS,
    'Handwriting': pb2.LayoutElement.ElementType.HANDWRITING
}
