from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import voca_service
from app.services.voca_service import VocaBookResult, VocaWordImageResult

router = APIRouter()



class VocaRequest(BaseModel):
    title: str
    category: str
    description: str
    numberOfDays: int

class WordItem(BaseModel):
    wordId: int
    term: str

class VocaWordImageRequest(BaseModel):
    words: list[WordItem]



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

@router.post("/voca/word/image", response_model=VocaWordImageResult)
async def generate_voca_word_image(request: VocaWordImageRequest) -> VocaWordImageResult:
    try:
        return await voca_service.generate_word_images(request.words)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
