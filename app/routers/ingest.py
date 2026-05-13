from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from app.services import ingest_service
from app.services.ingest_service import SUPPORTED_EXTENSIONS

router = APIRouter()


class IngestRequest(BaseModel):
    files: list[str]


@router.post("/admin/ingest/voca", status_code=202)
async def ingest_voca(request: IngestRequest, background_tasks: BackgroundTasks):
    if not request.files:
        raise HTTPException(status_code=400, detail="files가 비어있습니다.")

    try:
        ingest_service._validate_files(request.files)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(ingest_service.ingest_voca, request.files)
    return {"message": f"{len(request.files)}개 파일 인제스션이 시작되었습니다."}

