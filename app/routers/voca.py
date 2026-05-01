from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import voca_service
from app.services.voca_service import VocaBookResult

router = APIRouter()


class VocaRequest(BaseModel):
    title: str
    category: str
    description: str
    numberOfDays: int


@router.post("/voca", response_model=VocaBookResult)
async def generate_voca(request: VocaRequest) -> VocaBookResult:
    try:
        return await voca_service.generate_voca(
            request.title,
            request.category,
            request.description,
            request.numberOfDays,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
