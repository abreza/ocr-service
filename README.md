# OCR Service

This project implements a gRPC-based ...

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- gRPC-based ocr service
- Docker support for easy deployment
- Efficient ONNX runtime for inference

## Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/abreza/ocr-service.git
   cd ocr-service
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Generate gRPC code from the proto file:
   ```
   python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ocr_service.proto
   ```

## Usage

To start the gRPC server:

```
python main.py
```

The server will start and listen on port 50051.

## API Reference


## Docker Deployment

To build and run the service using Docker:

1. Build the Docker image:
   ```
   docker-compose build
   ```

2. Start the service:
   ```
   docker-compose up
   ```

The service will be available on the host machine at `localhost:50052`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
