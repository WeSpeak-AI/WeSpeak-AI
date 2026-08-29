import grpc

from app.grpc.chat_servicer import ChatServicer
from app.grpc.feedback_servicer import FeedbackServicer
from app.grpc.generated import wespeak_ai_pb2_grpc
from app.logger import get_logger

logger = get_logger("AI.grpc")

GRPC_PORT = 50051


async def create_grpc_server() -> grpc.aio.Server:
    server = grpc.aio.server()
    wespeak_ai_pb2_grpc.add_ChatServiceServicer_to_server(ChatServicer(), server)
    wespeak_ai_pb2_grpc.add_FeedbackServiceServicer_to_server(FeedbackServicer(), server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    await server.start()
    logger.info("gRPC server started on port %d", GRPC_PORT)
    return server
