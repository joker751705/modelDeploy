import uvicorn
from app.api.main import app

if __name__ == "__main__":
    # 运行Uvicorn服务器
    # 在生产环境中，推荐使用Gunicorn + Uvicorn worker
    uvicorn.run(app, host="127.0.0.1", port=8000)