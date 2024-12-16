import grpc
from typing import List, Tuple

from surya.layout import batch_layout_detection
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor

import ocr_service_pb2 as pb2
import ocr_service_pb2_grpc as pb2_grpc
from utils import create_detected_element, load_image, LABEL_TO_TYPE

class DocumentAnalysisServicer(pb2_grpc.DocumentAnalysisServicer):
    def __init__(self):
        self.layout_model = load_layout_model()
        self.layout_processor = load_layout_processor()

    def _process_layout_elements(self, layout_predictions) -> Tuple[List[pb2.LayoutElement], List[pb2.DetectedElement]]:
        """Process layout predictions and create layout elements with reading order."""
        layout_elements = []
        reading_order = []
        
        if hasattr(layout_predictions, 'bboxes') and layout_predictions.bboxes:
            try:
                bboxes = sorted(
                    [b for b in layout_predictions.bboxes if hasattr(b, 'bbox') and b.bbox],
                    key=lambda x: getattr(x, 'position', 0)
                )

                for bbox_data in bboxes:
                    if not hasattr(bbox_data, 'bbox') or not bbox_data.bbox:
                        continue
                        
                    layout_element = pb2.LayoutElement()
                    label = getattr(bbox_data, 'label', 'Text')
                    layout_element.type = LABEL_TO_TYPE.get(label, pb2.LayoutElement.ElementType.TEXT)

                    try:
                        element = create_detected_element(
                            bbox=bbox_data.bbox,
                            confidence=getattr(bbox_data, 'confidence', 1.0),
                            reading_order=getattr(bbox_data, 'position', 0)
                        )
                        
                        layout_element.element.CopyFrom(element)
                        layout_elements.append(layout_element)
                        reading_order.append(element)
                    except ValueError as e:
                        print(f"Skipping invalid bbox: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing layout elements: {str(e)}")

        return layout_elements, reading_order

    def AnalyzeDocument(self, request, context):
        try:
            image = load_image(request)
            response = pb2.DocumentAnalysisResponse()

            # Get layout analysis
            layout_predictions = batch_layout_detection(
                [image], 
                self.layout_model,
                self.layout_processor
            )[0]

            # Process layout elements
            layout_elements, reading_order = self._process_layout_elements(layout_predictions)
            response.layout_elements.extend(layout_elements)
            response.reading_order.extend(reading_order)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error analyzing document: {str(e)}')
            raise