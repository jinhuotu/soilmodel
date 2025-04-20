# src/middleware/logging_middleware.py
import time
import json
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from src.api.utils.logger import logger
from src.config import settings


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """增强型请求日志中间件"""

    def __init__(self, app: ASGIApp, *, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or []

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过健康检查等端点
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # 生成请求ID
        request_id = request.headers.get('X-Request-ID') or self._generate_request_id()

        # 记录开始时间
        start_time = time.time()
        response = None
        exception = None

        try:
            # 处理请求
            response = await call_next(request)
            return response
        except Exception as e:
            exception = e
            raise
        finally:
            # 计算耗时
            process_time = round((time.time() - start_time) * 1000, 2)

            # 构建日志数据
            log_data = self._build_log_data(
                request=request,
                response=response,
                exception=exception,
                process_time=process_time,
                request_id=request_id
            )

            # 记录日志
            self._log_request(log_data)

    def _build_log_data(
            self,
            request: Request,
            response: Optional[Response],
            exception: Optional[Exception],
            process_time: float,
            request_id: str
    ) -> Dict[str, Any]:
        """构建结构化日志数据"""
        return {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "status_code": response.status_code if response else 500,
            "process_time": f"{process_time}ms",
            "error": str(exception) if exception else None,
            "request_size": int(request.headers.get("content-length", 0)),
            "response_size": self._get_response_size(response),
            "user": self._get_authenticated_user(request)
        }

    def _log_request(self, log_data: Dict[str, Any]):
        """根据状态码选择日志级别"""
        if 500 <= log_data["status_code"] < 600:
            logger.error("Server Error", extra=log_data)
        elif 400 <= log_data["status_code"] < 500:
            logger.warning("Client Error", extra=log_data)
        else:
            logger.info("Request Completed", extra=log_data)

    def _generate_request_id(self) -> str:
        """生成UUID格式的请求ID"""
        import uuid
        return str(uuid.uuid4())

    def _get_response_size(self, response: Response) -> int:
        """获取响应体大小"""
        if not response or not response.body:
            return 0
        return len(response.body)

    def _get_authenticated_user(self, request: Request) -> Optional[str]:
        """获取认证用户信息"""
        user = request.state.user if hasattr(request.state, "user") else None
        return user.username if user else None

    def _redact_sensitive_data(self, data: dict) -> dict:
        """过滤敏感信息"""
        sensitive_keys = ["password", "token", "authorization"]
        return {
            k: "***REDACTED***" if k.lower() in sensitive_keys else v
            for k, v in data.items()
        }


# FastAPI 应用注册
def register_logging_middleware(app: ASGIApp):
    """注册日志中间件到应用"""
    excluded_paths = [
        "/health",
        "/favicon.ico",
        "/docs",
        "/openapi.json"
    ]
    app.add_middleware(
        RequestLoggingMiddleware,
        exclude_paths=excluded_paths
    )