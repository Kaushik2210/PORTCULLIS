"""PORTCULLIS gateway — OpenAI-compatible proxy and standalone detection API"""

from .app import create_app
from .config import GatewayConfig
from .pipeline import DetectionPipeline, DetectionResult

__all__ = ["DetectionPipeline", "DetectionResult", "GatewayConfig", "create_app"]
