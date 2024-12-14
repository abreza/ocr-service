import grpc
from concurrent import futures
import io
from PIL import Image

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


class OCRServicer(pb2_grpc.TextRecognitionServicer,
                  pb2_grpc.TextLineDetectionServicer,
                  pb2_grpc.LayoutAnalysisServicer,
                  pb2_grpc.ReadingOrderDetectionServicer,
                  pb2_grpc.TableRecognitionServicer):

    def __init__(self):
        self.det_model = load_det_model()
        self.det_processor = load_det_processor()
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()
        self.layout_model = load_layout_model()
        self.layout_processor = load_layout_processor()

    def _load_image(self, request):
        image_bytes = io.BytesIO(request.image_data)
        return Image.open(image_bytes)

    def _create_detected_element(self, bbox, polygon=None, content="", confidence=1.0):
        element = pb2.DetectedElement()
        element.content = content

        # Handle case where bbox might be a list of lists or nested structure
        if isinstance(bbox[0], (list, tuple)):
            bbox = [coord for sublist in bbox for coord in sublist][:4]  # Flatten and take first 4 coordinates
        
        # Ensure we have exactly 4 coordinates for bbox
        if len(bbox) >= 4:
            element.bbox.x1 = float(bbox[0])
            element.bbox.y1 = float(bbox[1])
            element.bbox.x2 = float(bbox[2])
            element.bbox.y2 = float(bbox[3])
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")

        # Handle polygon points if provided
        if polygon is not None:
            # Flatten polygon if it's nested
            if isinstance(polygon[0], (list, tuple)):
                polygon = [coord for sublist in polygon for coord in sublist]
            
            # Ensure even number of coordinates
            if len(polygon) % 2 == 0:
                element.polygon.x.extend([float(x) for x in polygon[::2]])
                element.polygon.y.extend([float(y) for y in polygon[1::2]])
            else:
                raise ValueError(f"Invalid polygon format: {polygon}")

        element.confidence = float(confidence)
        return element

    def RecognizeText(self, request, context):
        try:
            image = self._load_image(request)
            predictions = run_ocr([image], [["en", "fa"]], self.det_model,
                                self.det_processor, self.rec_model,
                                self.rec_processor)[0]

            response = pb2.TextRecognitionResponse()
            for line in predictions.text_lines:
                element = self._create_detected_element(
                    bbox=line.bbox,
                    polygon=line.polygon if hasattr(line, 'polygon') else None,
                    content=line.text,
                    confidence=line.confidence
                )
                response.text_lines.append(element)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error processing image: {str(e)}')
            raise

    def DetectTextLines(self, request, context):
        try:
            image = self._load_image(request)
            predictions = batch_text_detection([image], self.det_model,
                                               self.det_processor)[0]

            response = pb2.TextLineDetectionResponse()

            for bbox_data in predictions.get('bboxes', []):
                # Ensure bbox is in the correct format
                bbox = bbox_data['bbox']
                if isinstance(bbox[0], (list, tuple)):
                    bbox = [coord for sublist in bbox for coord in sublist][:4]

                element = self._create_detected_element(
                    bbox=bbox,
                    polygon=bbox_data.get('polygon', None),
                    confidence=bbox_data.get('confidence', 1.0)
                )
                response.text_lines.append(element)

            for line in predictions.get('vertical_lines', []):
                element = self._create_detected_element(
                    bbox=line['bbox']
                )
                response.vertical_lines.append(element)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error detecting text lines: {str(e)}')
            raise

    def AnalyzeLayout(self, request, context):
        try:
            image = self._load_image(request)

            line_predictions = batch_text_detection([image], self.det_model,
                                                    self.det_processor)[0]

            layout_predictions = batch_layout_detection([image], self.layout_model,
                                                        self.layout_processor,
                                                        [line_predictions])[0]

            response = pb2.LayoutAnalysisResponse()

            label_to_type = {
                'Page-header': pb2.LayoutElement.ElementType.HEADER,
                'Page-footer': pb2.LayoutElement.ElementType.FOOTER,
                'Table': pb2.LayoutElement.ElementType.TABLE,
                'Picture': pb2.LayoutElement.ElementType.IMAGE,
                'Caption': pb2.LayoutElement.ElementType.CAPTION,
                'Footnote': pb2.LayoutElement.ElementType.FOOTNOTE,
                'Formula': pb2.LayoutElement.ElementType.FORMULA,
                'List-item': pb2.LayoutElement.ElementType.LIST,
                'Section-header': pb2.LayoutElement.ElementType.SECTION_HEADER,
                'Form': pb2.LayoutElement.ElementType.FORM,
                'Table-of-contents': pb2.LayoutElement.ElementType.TOC,
                'Handwriting': pb2.LayoutElement.ElementType.HANDWRITING,
                'Text-inline-math': pb2.LayoutElement.ElementType.INLINE_MATH
            }

            for bbox_data in layout_predictions.get('bboxes', []):
                layout_element = pb2.LayoutElement()
                layout_element.type = label_to_type.get(
                    bbox_data.get('label', 'Text'),
                    pb2.LayoutElement.ElementType.HEADER
                )

                # Ensure bbox is in the correct format
                bbox = bbox_data['bbox']
                if isinstance(bbox[0], (list, tuple)):
                    bbox = [coord for sublist in bbox for coord in sublist][:4]

                element = self._create_detected_element(
                    bbox=bbox,
                    polygon=bbox_data.get('polygon', None)
                )
                layout_element.element.CopyFrom(element)

                response.elements.append(layout_element)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error analyzing layout: {str(e)}')
            raise

    def DetectReadingOrder(self, request, context):
        try:
            image = self._load_image(request)

            line_predictions = batch_text_detection([image], self.det_model,
                                                    self.det_processor)[0]

            layout_predictions = batch_layout_detection([image], self.layout_model,
                                                        self.layout_processor,
                                                        [line_predictions])[0]

            response = pb2.ReadingOrderResponse()

            sorted_bboxes = sorted(
                layout_predictions.get('bboxes', []),
                key=lambda x: x.get('position', 0)
            )

            for bbox_data in sorted_bboxes:
                text_block = pb2.TextBlock()

                bbox = bbox_data['bbox']
                print('1: ', bbox)
                if isinstance(bbox[0], (list, tuple)):
                    bbox = [coord for sublist in bbox for coord in sublist][:4]
                print('2: ', bbox)

                element = self._create_detected_element(
                    bbox=bbox,
                    polygon=bbox_data.get('polygon', None)
                )
                text_block.element.CopyFrom(element)
                text_block.position = bbox_data.get('position', 0)

                response.blocks.append(text_block)

            return response

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error detecting reading order: {str(e)}')
            raise


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = OCRServicer()

    pb2_grpc.add_TextRecognitionServicer_to_server(servicer, server)
    pb2_grpc.add_TextLineDetectionServicer_to_server(servicer, server)
    pb2_grpc.add_LayoutAnalysisServicer_to_server(servicer, server)
    pb2_grpc.add_ReadingOrderDetectionServicer_to_server(servicer, server)
    pb2_grpc.add_TableRecognitionServicer_to_server(servicer, server)

    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
