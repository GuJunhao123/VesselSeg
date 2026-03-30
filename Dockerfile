# 使用官方Python 3.9镜像作为基础
FROM python:3.9-slim-buster

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建输出目录
RUN mkdir -p output

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/model

# 暴露端口（如果需要）
# EXPOSE 8000

# 运行应用
CMD ["python", "main.py"]