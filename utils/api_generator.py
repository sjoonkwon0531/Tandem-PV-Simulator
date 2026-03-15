"""
API Generator: OpenAPI Spec Generation
=======================================

Generate RESTful API documentation (OpenAPI/Swagger) from the Streamlit app.

Features:
- Auto-generate OpenAPI 3.0 spec
- Predict bandgap from composition endpoint
- Batch prediction endpoint
- Rate limiting simulation (in-memory)
- Usage tracking

Note: This generates the SPECIFICATION only (no actual FastAPI server).
For production, use the spec to implement FastAPI endpoints.

Author: OpenClaw Agent
Date: 2026-03-15 (V8)
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd


class APISpecGenerator:
    """
    Generate OpenAPI 3.0 specification for the Tandem PV API.
    
    Spec includes:
    - /predict (single composition → bandgap prediction)
    - /predict/batch (multiple compositions → multiple predictions)
    - /models/list (list available models)
    - /health (service health check)
    """
    
    def __init__(self, title: str = "AlphaMaterials API", version: str = "8.0.0"):
        """
        Initialize API spec generator.
        
        Args:
            title: API title
            version: API version
        """
        self.title = title
        self.version = version
    
    def generate_spec(self) -> Dict[str, Any]:
        """
        Generate complete OpenAPI 3.0 specification.
        
        Returns:
            OpenAPI spec as dictionary (ready for JSON export)
        """
        
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": self.title,
                "version": self.version,
                "description": "Machine learning API for perovskite bandgap prediction and materials discovery.",
                "contact": {
                    "name": "AlphaMaterials Team",
                    "email": "support@alphamaterials.ai"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "https://api.alphamaterials.ai/v1",
                    "description": "Production server"
                },
                {
                    "url": "http://localhost:8000/v1",
                    "description": "Development server"
                }
            ],
            "paths": self._generate_paths(),
            "components": self._generate_components()
        }
        
        return spec
    
    def _generate_paths(self) -> Dict[str, Any]:
        """Generate API endpoint definitions."""
        
        return {
            "/predict": {
                "post": {
                    "summary": "Predict bandgap for single composition",
                    "description": "Predict the bandgap of a perovskite composition using the trained ML model.",
                    "operationId": "predictBandgap",
                    "tags": ["Prediction"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/PredictRequest"
                                },
                                "example": {
                                    "composition": "MAPbI3",
                                    "model_id": "halide-base-v1"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful prediction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/PredictResponse"
                                    },
                                    "example": {
                                        "composition": "MAPbI3",
                                        "bandgap": 1.59,
                                        "confidence": 0.95,
                                        "model_id": "halide-base-v1",
                                        "timestamp": "2026-03-15T12:00:00Z"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Invalid composition format",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        },
                        "429": {
                            "description": "Rate limit exceeded",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict/batch": {
                "post": {
                    "summary": "Predict bandgaps for multiple compositions",
                    "description": "Batch prediction endpoint for multiple compositions.",
                    "operationId": "predictBatch",
                    "tags": ["Prediction"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/BatchPredictRequest"
                                },
                                "example": {
                                    "compositions": ["MAPbI3", "FAPbI3", "CsPbI3"],
                                    "model_id": "halide-base-v1"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Successful batch prediction",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/BatchPredictResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models": {
                "get": {
                    "summary": "List available models",
                    "description": "Get list of all trained models in the model zoo.",
                    "operationId": "listModels",
                    "tags": ["Models"],
                    "parameters": [
                        {
                            "name": "family",
                            "in": "query",
                            "description": "Filter by model family",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["base", "fine-tuned", "domain-specific", "user-trained"]
                            }
                        },
                        {
                            "name": "domain",
                            "in": "query",
                            "description": "Filter by material domain",
                            "required": False,
                            "schema": {
                                "type": "string",
                                "enum": ["halide_perovskites", "oxide_perovskites", "chalcogenides", "general"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ModelListResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "description": "Check API service health and status.",
                    "operationId": "healthCheck",
                    "tags": ["System"],
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    },
                                    "example": {
                                        "status": "healthy",
                                        "version": "8.0.0",
                                        "timestamp": "2026-03-15T12:00:00Z",
                                        "models_loaded": 5,
                                        "uptime_seconds": 86400
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def _generate_components(self) -> Dict[str, Any]:
        """Generate reusable schema components."""
        
        return {
            "schemas": {
                "PredictRequest": {
                    "type": "object",
                    "required": ["composition"],
                    "properties": {
                        "composition": {
                            "type": "string",
                            "description": "Chemical formula (e.g., MAPbI3, MA0.5FA0.5PbI3)",
                            "example": "MAPbI3"
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model ID to use (default: latest base model)",
                            "example": "halide-base-v1"
                        }
                    }
                },
                "PredictResponse": {
                    "type": "object",
                    "properties": {
                        "composition": {
                            "type": "string",
                            "description": "Input composition"
                        },
                        "bandgap": {
                            "type": "number",
                            "format": "float",
                            "description": "Predicted bandgap (eV)"
                        },
                        "confidence": {
                            "type": "number",
                            "format": "float",
                            "description": "Prediction confidence (0-1)"
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model used for prediction"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Prediction timestamp"
                        }
                    }
                },
                "BatchPredictRequest": {
                    "type": "object",
                    "required": ["compositions"],
                    "properties": {
                        "compositions": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of chemical formulas",
                            "example": ["MAPbI3", "FAPbI3", "CsPbI3"]
                        },
                        "model_id": {
                            "type": "string",
                            "description": "Model ID to use"
                        }
                    }
                },
                "BatchPredictResponse": {
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/PredictResponse"
                            }
                        },
                        "total": {
                            "type": "integer",
                            "description": "Total predictions"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                },
                "ModelListResponse": {
                    "type": "object",
                    "properties": {
                        "models": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "model_id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "version": {"type": "string"},
                                    "family": {"type": "string"},
                                    "domain": {"type": "string"},
                                    "mae": {"type": "number"},
                                    "r2": {"type": "number"}
                                }
                            }
                        },
                        "total": {
                            "type": "integer"
                        }
                    }
                },
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["healthy", "degraded", "down"]
                        },
                        "version": {
                            "type": "string"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "models_loaded": {
                            "type": "integer"
                        },
                        "uptime_seconds": {
                            "type": "integer"
                        }
                    }
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "string",
                            "description": "Error message"
                        },
                        "code": {
                            "type": "string",
                            "description": "Error code"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    }
                }
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key for authentication"
                }
            }
        }
    
    def export_json(self, filepath: str):
        """
        Export OpenAPI spec to JSON file.
        
        Args:
            filepath: Output file path
        """
        spec = self.generate_spec()
        
        with open(filepath, 'w') as f:
            json.dump(spec, f, indent=2)
    
    def export_yaml(self, filepath: str):
        """
        Export OpenAPI spec to YAML file (requires PyYAML).
        
        Args:
            filepath: Output file path
        """
        try:
            import yaml
            spec = self.generate_spec()
            
            with open(filepath, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            raise ImportError("PyYAML not installed. Use export_json() instead.")


class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    Tracks API calls per client and enforces limits.
    """
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is allowed to make a request.
        
        Args:
            client_id: Client identifier (IP, API key, etc.)
        
        Returns:
            True if allowed, False if rate limit exceeded
        """
        now = datetime.now().timestamp()
        cutoff = now - self.window_seconds
        
        # Remove old requests
        self.requests[client_id] = [
            ts for ts in self.requests[client_id]
            if ts > cutoff
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True
    
    def get_stats(self, client_id: str) -> Dict[str, Any]:
        """
        Get rate limit statistics for a client.
        
        Args:
            client_id: Client identifier
        
        Returns:
            Statistics dictionary
        """
        now = datetime.now().timestamp()
        cutoff = now - self.window_seconds
        
        recent = [ts for ts in self.requests[client_id] if ts > cutoff]
        
        return {
            'requests_in_window': len(recent),
            'max_requests': self.max_requests,
            'window_seconds': self.window_seconds,
            'remaining': max(0, self.max_requests - len(recent)),
            'reset_at': cutoff + self.window_seconds
        }


class UsageTracker:
    """
    Track API usage statistics.
    """
    
    def __init__(self):
        """Initialize usage tracker."""
        self.total_requests = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.total_compositions_predicted = 0
        
        self.endpoint_counts = defaultdict(int)
        self.model_usage = defaultdict(int)
        
        self.start_time = datetime.now()
    
    def record_request(self, endpoint: str, success: bool, model_id: Optional[str] = None, count: int = 1):
        """
        Record API request.
        
        Args:
            endpoint: Endpoint called
            success: Whether request succeeded
            model_id: Model used (if applicable)
            count: Number of predictions (for batch)
        """
        self.total_requests += 1
        self.endpoint_counts[endpoint] += 1
        
        if success:
            self.successful_predictions += count
            self.total_compositions_predicted += count
            
            if model_id:
                self.model_usage[model_id] += count
        else:
            self.failed_predictions += count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Statistics dictionary
        """
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'total_requests': self.total_requests,
            'successful_predictions': self.successful_predictions,
            'failed_predictions': self.failed_predictions,
            'total_compositions': self.total_compositions_predicted,
            'success_rate': self.successful_predictions / max(1, self.total_requests),
            'uptime_seconds': uptime,
            'requests_per_second': self.total_requests / max(1, uptime),
            'endpoint_counts': dict(self.endpoint_counts),
            'model_usage': dict(self.model_usage),
            'most_used_model': max(self.model_usage.items(), key=lambda x: x[1])[0] if self.model_usage else None
        }
