
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import uuid
import re
import os
from datetime import datetime
from pathlib import Path

from ...core.classification import EvaluationTypeClassifier
from ...utils.text_processing import create_smart_chunks_from_files
from ...utils.logging import get_logger
from ...utils.config import ChunkingConfig

router = APIRouter()
logger = get_logger("corpus_routes")

class CorpusUploadRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10*1024*1024, description="Corpus text content")
    name: Optional[str] = Field("corpus", max_length=100, description="Corpus name")
    description: Optional[str] = Field("", max_length=500, description="Corpus description")

    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty")
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:.*base64',
            r'\\x[0-9a-fA-F]{2}',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                raise ValueError("Text contains potentially malicious content")
        return v.strip()

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v:
            sanitized = re.sub(r'[^\w\s\-_\(\)]', '', v)
            sanitized = re.sub(r'\s+', ' ', sanitized).strip()
            if len(sanitized) < 3:
                sanitized = "Uploaded Corpus"
            return sanitized
        return v.strip() if v else v

@router.post("/upload")
async def upload_corpus_text(request: CorpusUploadRequest):
    """Upload corpus text for analysis"""
    try:
        classifier = EvaluationTypeClassifier()
        classification = classifier.classify_corpus(request.text)
        corpus_id = str(uuid.uuid4())
        corpus_info = {
            "id": corpus_id,
            "name": request.name,
            "description": request.description,
            "text": request.text,
            "classification": classification.to_dict(),
            "stats": {
                "characters": len(request.text),
                "words": len(request.text.split()),
                "lines": len(request.text.splitlines())
            },
            "created_at": datetime.now().isoformat()
        }
        logger.info("Corpus uploaded", corpus_id=corpus_id, chars=len(request.text), primary_type=classification.primary_type)
        return {
            "corpus_id": corpus_id,
            "classification": classification.to_dict(),
            "stats": corpus_info["stats"],
            "corpus_text": request.text
        }
    except Exception as e:
        logger.error(f"Error uploading corpus: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-file")
async def upload_corpus_file(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Upload corpus from file with validation"""
    try:
        MAX_FILE_SIZE = 10 * 1024 * 1024
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large: {len(content)} bytes exceeds limit")
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a name")
        allowed_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml', '.yml', '.yaml', '.cfg', '.ini', '.log'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"File type {file_ext} not allowed")
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('latin-1')
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Cannot decode file as text")
        MAX_TEXT_LENGTH = 5 * 1024 * 1024
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=413, detail="Text content too long")
        suspicious_patterns = [r'<script[^>]*>.*?</script>', r'javascript:', r'data:.*base64', r'\\x[0-9a-fA-F]{2}']
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(status_code=400, detail="File contains suspicious content")
        request = CorpusUploadRequest(
            text=text,
            name=name or file.filename or "uploaded_corpus",
            description=f"Uploaded from file: {file.filename}"
        )
        return await upload_corpus_text(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/upload-multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    name: Optional[str] = Form(None),
    min_tokens: Optional[int] = Form(None),
    target_tokens: Optional[int] = Form(None),
    max_tokens: Optional[int] = Form(None),
    overlap_tokens: Optional[int] = Form(None),
):
    """Upload multiple files as a single corpus with smart chunking"""
    try:
        file_contents = []
        file_names = []
        total_size = 0
        supported_extensions = {'.txt', '.md', '.py', '.js', '.json', '.csv', '.html', '.xml', '.yml', '.yaml', '.cfg', '.ini', '.log'}
        for file in files:
            if not file.filename:
                continue
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in supported_extensions:
                continue
            try:
                content = await file.read()
                total_size += len(content)
                if total_size > 10 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Total file size exceeds 10MB limit")
                try:
                    text = content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        text = content.decode('latin-1')
                    except UnicodeDecodeError:
                        continue
                file_contents.append({'filename': file.filename, 'content': text.strip()})
                file_names.append(file.filename)
            except Exception:
                continue
        if not file_contents:
            raise HTTPException(status_code=400, detail="No valid text files found")
        disable_chonkie = os.getenv("DISABLE_CHONKIE", "false").lower() in ["true", "1", "yes", "on"]
        chunking_config = ChunkingConfig(use_token_chunking=True, target_token_size=3000, max_token_size=4000, enable_chonkie=not disable_chonkie)
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))
        if isinstance(min_tokens, int):
            chunking_config.min_token_size = clamp(min_tokens, 500, 8192)
        if isinstance(target_tokens, int):
            chunking_config.target_token_size = clamp(target_tokens, 500, 8192)
        if isinstance(max_tokens, int):
            chunking_config.max_token_size = clamp(max_tokens, 500, 8192)
        if isinstance(overlap_tokens, int):
            chunking_config.overlap_tokens = clamp(overlap_tokens, 0, 1024)
        chunks = create_smart_chunks_from_files(file_contents, chunking_config)
        combined_text = ""
        chunk_info = []
        for i, chunk in enumerate(chunks):
            if combined_text:
                combined_text += f"\n\n{'='*80}\nCHUNK {i+1} (from {len(chunk['file_sources'])} files)\n{'='*80}\n\n"
            combined_text += chunk['text']
            chunk_info.append({'chunk_index': i, 'size_chars': chunk['size'], 'file_count': len(chunk['file_sources']), 'method': chunk['method']})
        corpus_name = name or f"Smart-chunked ({len(file_names)} files)"
        request = CorpusUploadRequest(text=combined_text, name=corpus_name, description=f"Smart-chunked corpus from {len(file_names)} files")
        result = await upload_corpus_text(request)
        result.update({"files_processed": len(file_names), "chunks_created": len(chunks), "smart_chunking_enabled": True})
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading multiple: {e}")
        raise HTTPException(status_code=500, detail=str(e))
