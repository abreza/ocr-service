import io
from typing import List
from PIL import Image
import ocr_service_pb2 as pb2

def load_image(request) -> Image.Image:
    """Load image from request bytes."""
    image_bytes = io.BytesIO(request.image_data)
    return Image.open(image_bytes)

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

# Map Surya labels to proto enum types
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