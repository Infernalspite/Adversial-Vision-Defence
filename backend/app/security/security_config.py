"""Environment-backed security and resource limits."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class SecuritySettings(BaseSettings):
    max_image_size_mb: float = 10.0
    max_image_width: int = 4096
    max_image_height: int = 4096
    allowed_image_formats: str = "jpg,jpeg,png,webp"
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    request_timeout_seconds: float = 120.0
    max_attack_iterations: int = 100
    max_patch_size: float = 1.0
    max_evolution_generations: int = 10
    max_population_size: int = 100
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    audit_hashing_enabled: bool = True
    demo_mode: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ARGUS_", extra="ignore")

    @property
    def max_image_bytes(self) -> int:
        return int(self.max_image_size_mb * 1024 * 1024)

    @property
    def formats(self) -> set[str]:
        return {item.strip().lower() for item in self.allowed_image_formats.split(",") if item.strip()}

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]


security_settings = SecuritySettings()
