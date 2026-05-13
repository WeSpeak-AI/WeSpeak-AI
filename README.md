# WeSpeak AI Server

LangChain + Ollama 기반 영어 학습 AI API 서버입니다.

## 기술 스택

- **Framework**: FastAPI
- **LLM**: Ollama (llama3.2) / Claude (claude-haiku)
- **STT**: faster-whisper (medium / large-v3)
- **Embedding**: Upstage Solar Embedding
- **Vector DB**: Pinecone
- **GPU**: NVIDIA CUDA

## 프로젝트 구조

```
WeSpeak-AI/
├── app/
│   ├── main.py               # FastAPI 앱 진입점, 미들웨어
│   ├── config.py             # 환경변수 설정
│   ├── logger.py             # 로깅 설정
│   ├── routers/
│   │   ├── chat.py           # 채팅
│   │   ├── feedback.py       # 음성 요약 피드백
│   │   ├── search.py         # 단어 검색
│   │   ├── correct.py        # 문장 교정
│   │   ├── topic.py          # 토픽 생성
│   │   ├── voca.py           # 단어장 생성
│   │   └── ingest.py         # 파일 임베딩 (Admin)
│   └── services/
│       ├── llm.py            # LLM 인스턴스 관리
│       ├── stt_service.py    # 음성 → 텍스트 (faster-whisper)
│       ├── tts_service.py    # 텍스트 → 음성
│       ├── chat_service.py   # 채팅 로직
│       ├── feedback_service.py  # 피드백 로직
│       ├── search_service.py    # 단어 검색 로직
│       ├── correction_service.py # 문장 교정 로직
│       ├── topic_service.py     # 토픽 생성 로직
│       ├── voca_service.py      # 단어장 생성 로직 (RAG)
│       └── ingest_service.py    # 파일 임베딩 로직
└── scripts/
    └── ingest_voca.py        # CLI 파일 임베딩 스크립트
```

## API 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 서버 상태 확인 |
| POST | `/chat` | 영어 채팅 |
| POST | `/feedback` | 음성 요약 피드백 |
| POST | `/search` | 단어 검색 |
| POST | `/correct` | 문장 교정 |
| POST | `/topic` | 토픽 생성 |
| POST | `/voca` | 단어장 생성 |
| POST | `/ingest` | 파일 임베딩 (Admin) |

## 환경변수 설정

`.env.example`을 복사해서 `.env`를 생성합니다.

```bash
cp .env.example .env
```

| 변수 | 설명 | 기본값 |
|---|---|---|
| `PROVIDER` | LLM 제공자 (`ollama` / `claude`) | `ollama` |
| `OLLAMA_BASE_URL` | Ollama 서버 주소 | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama 모델명 | `llama3.2` |
| `ANTHROPIC_API_KEY` | Claude API 키 | - |
| `CLAUDE_MODEL` | Claude 모델명 | `claude-haiku-4-5-20251001` |
| `PINECONE_API_KEY` | Pinecone API 키 | - |
| `PINECONE_VOCA_INDEX` | Pinecone 인덱스명 | `voca-index` |
| `UPSTAGE_API_KEY` | Upstage API 키 | - |
| `WHISPER_MODEL` | Whisper 모델 크기 | `medium` |
| `WHISPER_DEVICE` | 실행 디바이스 | `cuda` |
| `WHISPER_COMPUTE_TYPE` | 연산 정밀도 | `int8` |

## 실행 방법

### Docker (권장)

```bash
docker compose up -d
```

### 로컬 실행

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 파일 임베딩 (CLI)

voca-index에 파일을 임베딩합니다. 기존 데이터는 삭제 후 새로 삽입됩니다.

```bash
python scripts/ingest_voca.py assets/books/Basic/A\ Christmas\ Carol.txt
```

지원 형식: `.txt`, `.pdf`, `.docx`
