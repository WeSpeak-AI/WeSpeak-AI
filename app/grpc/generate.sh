#!/usr/bin/env bash
# wespeak_ai.proto로부터 Python gRPC 코드를 생성한다.
# 사용법: (repo root에서) ./app/grpc/generate.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

python -m grpc_tools.protoc \
  -I app/grpc/protos \
  --python_out=app/grpc/generated \
  --grpc_python_out=app/grpc/generated \
  --pyi_out=app/grpc/generated \
  app/grpc/protos/wespeak_ai.proto

# grpc_tools.protoc가 생성하는 *_pb2_grpc.py의 import가 절대경로(import wespeak_ai_pb2)라
# 패키지 상대 import로 고쳐준다.
sed -i.bak 's/^import wespeak_ai_pb2/from . import wespeak_ai_pb2/' app/grpc/generated/wespeak_ai_pb2_grpc.py
rm -f app/grpc/generated/wespeak_ai_pb2_grpc.py.bak

echo "generated: app/grpc/generated/wespeak_ai_pb2.py, wespeak_ai_pb2_grpc.py"
