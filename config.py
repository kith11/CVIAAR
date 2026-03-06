from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Core App Settings
    APP_ROLE: str = "LOCAL_KIOSK"
    DEVICE_ID: str = "local-device"
    SECRET_KEY: str = "a-very-secret-key-that-you-should-change"
    ADMIN_PASSWORD: str = "admin123"

    # Database URLs
    SQLITE_DB_PATH: str = "data/offline/cviaar_local.sqlite3"
    DATABASE_URL: str | None = None # For Supabase/Postgres

    # Upstash Redis for Live Layer
    UPSTASH_REDIS_URL: str
    UPSTASH_REDIS_TOKEN: str

    # Face Recognition Tunables
    LBPH_DISTANCE_THRESHOLD: float = 95.0
    RECOGNITION_CACHE_TTL_SEC: float = 2.0
    BLINK_EAR_THRESHOLD: float = 0.20
    BLINK_MAX_CLOSED_SEC: float = 2.5
    VERIFIED_TTL_SEC: float = 20.0

    # Hugging Face API for AI Chatbot
    HF_TOKEN: str | None = None

settings = Settings()
