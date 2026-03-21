# our_snap_new/config.py
import os

class Config:
    # 1. 基础路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(BASE_DIR, "saved_records") # 图片存档路径
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 2. 数据库连接配置 (最关键的部分)
    # 驱动：通常用 'ODBC Driver 17 for SQL Server'，如果报错改用 'SQL Server'
    DB_DRIVER = 'ODBC Driver 17 for SQL Server'
    
    # 服务器地址：这是你之前连成功的那个地址
    # 注意前面的 r 不能少
    DB_SERVER = r'LAPTOP-QNGAAJ92\SQLEXPRESS'
    
    # 数据库名：你之前建好的那个库
    DB_NAME = 'MedicalDB'

    # 认证方式：Windows 身份验证 (True)
    USE_WINDOWS_AUTH = True
    
    # 备用账号密码 (仅当 USE_WINDOWS_AUTH=False 时生效)
    DB_USER = 'sa'
    DB_PASSWORD = 'your_password'