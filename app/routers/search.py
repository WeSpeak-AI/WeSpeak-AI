from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import search_service

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    term: str
    meaning: str
    phonetic: str
    example: str


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    try:
        result = await search_service.search_word(request.query)
        return SearchResponse(
            term=result.term,
            meaning=result.meaning,
            phonetic=result.phonetic,
            example=result.example,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
