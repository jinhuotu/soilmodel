
from fastapi import FastAPI
from src.middleware.logging_middleware import register_logging_middleware

app = FastAPI()

# 注册中间件
register_logging_middleware(app)

# ...其他路由和配置...