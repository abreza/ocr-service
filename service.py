import grpc
from typing import List, Tuple

from surya.ocr import run_ocr
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.model.layout.model import load_model as load_layout_model
from surya.model.layout.processor import load_processor as load_layout_processor

import ocr_service_pb2 as pb2
import ocr_service_pb2_grpc as pb2_grpc
from utils import create_detected_element, load_image, LABEL_TO_TYPE


class DocumentAnalysisServicer(pb2_grpc.DocumentAnalysisServicer):
    def __init__(self):
        self.det_model = load_det_model()
        self.det_processor = load_det_processor()
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()
        self.layout_model = load_layout_model()
        self.layout_processor = load_layout_processor()

    def _process_ocr_results(self, ocr_predictions) -> List[pb2.DetectedElement]:
        """Process OCR predictions and convert to DetectedElement objects."""
        elements = []
        for line in ocr_predictions.text_lines:
            element = create_detected_element(
                bbox=line.bbox,
                polygon=line.polygon if hasattr(line, 'polygon') else None,
                content=line.text,
                confidence=line.confidence
            )
            elements.append(element)
        return elements

    def _process_vertical_lines(self, line_predictions) -> List[pb2.DetectedElement]:
        """Process vertical lines from detection results."""
        elements = []
        # Access vertical_lines as an attribute instead of using get()
        if hasattr(line_predictions, 'vertical_lines'):
            for line in line_predictions.vertical_lines:
                element = create_detected_element(bbox=line.bbox)
                elements.append(element)
        return elements

    def _process_layout_elements(self, layout_predictions) -> Tuple[List[pb2.LayoutElement], List[pb2.DetectedElement]]:
        """Process layout predictions and create layout elements with reading order."""
        layout_elements = []
        reading_order = []
        
        # Access bboxes directly as an attribute
        if hasattr(layout_predictions, 'bboxes'):
            # Sort bboxes by position if available
            bboxes = sorted(
                layout_predictions.bboxes,
                key=lambda x: getattr(x, 'position', 0)
            )

            for bbox_data in bboxes:
                layout_element = pb2.LayoutElement()
                # Get label safely using getattr
                label = getattr(bbox_data, 'label', 'Text')
                layout_element.type = LABEL_TO_TYPE.get(label, pb2.LayoutElement.ElementType.TEXT)

                element = create_detected_element(
                    bbox=bbox_data.bbox,
                    polygon=getattr(bbox_data, 'polygon', None),
                    reading_order=getattr(bbox_data, 'position', 0)
                )
                
                layout_element.element.CopyFrom(element)
                layout_elements.append(layout_element)
                reading_order.append(element)

        return layout_elements, reading_order

    def AnalyzeDocument(self, request, context):
        try:
            image = load_image(request)
            response = pb2.DocumentAnalysisResponse()

            # Run OCR
            ocr_predictions = run_ocr(
                [image], 
                [["en", "fa"]], 
                self.det_model,
                self.det_processor, 
                self.rec_model,
                self.rec_processor
            )[0]

            # Get text line detections
            line_predictions = batch_text_detection(
                [image], 
                self.det_model,
                self.det_processor
            )[0]

            # Get layout analysis
            layout_predictions = batch_layout_detection(
                [image], 
                self.layout_model,
                self.layout_processor,
                [line_predictions]
            )[0]

            # Process results
            response.text_lines.extend(self._process_ocr_results(ocr_predictions))
            response.vertical_lines.extend(self._process_vertical_lines(line_predictions))
            
            # Process layout elements
            layout_elements, reading_order = self._process_layout_elements(layout_predictions)
            response.layout_elements.extend(layout_elements)
            response.reading_order.extend(reading_order)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error analyzing document: {str(e)}')
            raise