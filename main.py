import grpc
from concurrent import futures
from service import DocumentAnalysisServicer
import ocr_service_pb2_grpc as pb2_grpc

def serve(port: int = 50051, max_workers: int = 10):
    """Start the gRPC server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = DocumentAnalysisServicer()
    pb2_grpc.add_DocumentAnalysisServicer_to_server(servicer, server)

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Server started on port {port}")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
