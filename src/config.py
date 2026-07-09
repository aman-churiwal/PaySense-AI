"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    gemini_api_key: str = ""
    chroma_persist_dir: str = "./chroma_data"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    gemini_model_name: str = "gemini-2.0-flash"
    upload_max_bytes: int = 10_485_760  # 10 MB
    supported_formats: list[str] = field(
        default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg"]
    )

    @classmethod
    def from_env(cls) -> "Settings":
        """Create Settings from environment variables."""
        load_dotenv()
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            gemini_model_name=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            upload_max_bytes=int(os.getenv("UPLOAD_MAX_BYTES", "10485760")),
        )
