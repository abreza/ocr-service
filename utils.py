import io
from typing import List, Union, Optional
from PIL import Image
import ocr_service_pb2 as pb2

def load_image(request) -> Image.Image:
    """Load image from request bytes."""
    image_bytes = io.BytesIO(request.image_data)
    return Image.open(image_bytes)

def create_detected_element(
    bbox: Union[List[float], object],
    polygon: Optional[Union[List[float], object]] = None,
    content: str = "",
    confidence: float = 1.0,
    reading_order: int = 0
) -> pb2.DetectedElement:
    """Create a DetectedElement from detection results."""
    element = pb2.DetectedElement()
    element.content = content

    # Handle bbox that might be an object with bbox attribute
    if hasattr(bbox, 'bbox'):
        bbox = bbox.bbox

    if isinstance(bbox[0], (list, tuple)):
        bbox = [coord for sublist in bbox for coord in sublist][:4]

    if len(bbox) >= 4:
        element.bbox.x1 = float(bbox[0])
        element.bbox.y1 = float(bbox[1])
        element.bbox.x2 = float(bbox[2])
        element.bbox.y2 = float(bbox[3])
    else:
        raise ValueError(f"Invalid bbox format: {bbox}")

    if polygon is not None:
        # Handle polygon that might be an object with polygon attribute
        if hasattr(polygon, 'polygon'):
            polygon = polygon.polygon

        if isinstance(polygon[0], (list, tuple)):
            polygon = [coord for sublist in polygon for coord in sublist]

        if len(polygon) % 2 == 0:
            element.polygon.x.extend([float(x) for x in polygon[::2]])
            element.polygon.y.extend([float(y) for y in polygon[1::2]])
        else:
            raise ValueError(f"Invalid polygon format: {polygon}")

    element.confidence = float(confidence)
    element.reading_order = reading_order
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