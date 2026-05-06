from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Core App Settings
    APP_ROLE: str = "LOCAL_KIOSK"
    DEVICE_ID: str = "local-device"
    DEBUG: bool = False
    SECRET_KEY: str = "a-very-secret-key-that-you-should-change"
    ADMIN_PASSWORD: str = "admin123"
    SESSION_MAX_AGE_SECONDS: int = 60 * 60 * 12

    # Database URLs
    SQLITE_DB_PATH: str = "data/offline/cviaar_local.sqlite3"
    DATABASE_URL: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL", "SUPABASE_DB_URL"),
    )
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    SYNC_ENABLED: bool = True # Enable or disable the sync worker

    # Upstash Redis for Live Layer
    UPSTASH_REDIS_URL: str | None = None
    UPSTASH_REDIS_TOKEN: str | None = None
    REDIS_URL: str | None = "redis://redis:6379/0" # Default to docker service

    # Face Recognition Tunables
    LBPH_DISTANCE_THRESHOLD: float = 95.0
    """Threshold for the LBPH face recognizer. Lower is stricter."""
    RECOGNITION_CACHE_TTL_SEC: float = 2.0
    """How long to cache a recognized face to avoid re-processing."""
    BLINK_EAR_THRESHOLD: float = 0.22
    """Eye Aspect Ratio threshold to trigger blink detection. A lower value makes it less sensitive to slight eye movements."""
    BLINK_MAX_CLOSED_SEC: float = 2.5
    """Maximum duration for eyes to be closed before resetting the blink detector."""
    VERIFIED_TTL_SEC: float = 6.0
    """How long a user remains 'verified' after a successful blink."""

    # Camera Settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 15

    # Camera Settings
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 15

    # Hugging Face API for AI Chatbot
    HF_TOKEN: str | None = None

    # Gmail Credentials for Report Sending
    MAIL_USERNAME: str | None = None
    MAIL_APP_PASSWORD: str | None = None

settings = Settings()
