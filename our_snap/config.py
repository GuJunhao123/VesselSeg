# our_snap_new/config.py
import os
from datetime import datetime

class Config:
    # 1. 基础路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "saved_records") # 图片存档路径
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 2. 日志文件配置
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    # 主日志文件路径
    MAIN_LOG_FILE = os.path.join(LOG_DIR, "main.log")
    
    # 病例记录日志文件路径
    CASE_LOG_FILE = os.path.join(LOG_DIR, "case_records.log")