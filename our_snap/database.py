# our_snap_new/database.py

import pyodbc
import os
import shutil
import traceback
from datetime import datetime
from config import Config

class DatabaseManager:
    def __init__(self):
        # 初始化时尝试建表，防止表不存在报错
        self.init_tables()

    def get_connection(self):
        """获取数据库连接"""
        if Config.USE_WINDOWS_AUTH:
             conn_str = f"DRIVER={{{Config.DB_DRIVER}}};SERVER={Config.DB_SERVER};DATABASE={Config.DB_NAME};Trusted_Connection=yes;"
        else:
             conn_str = f"DRIVER={{{Config.DB_DRIVER}}};SERVER={Config.DB_SERVER};DATABASE={Config.DB_NAME};UID={Config.DB_USER};PWD={Config.DB_PASSWORD};"
        try:
            return pyodbc.connect(conn_str)
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return None

    def init_tables(self):
        """初始化表结构，确保 segmentation_records 表存在"""
        conn = self.get_connection()
        if not conn: return
        cursor = conn.cursor()
        
        # 创建【影像记录表】
        sql = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='segmentation_records' AND xtype='U')
            BEGIN
                CREATE TABLE segmentation_records (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    patient_code NVARCHAR(50),      -- 关联病人编号
                    patient_name NVARCHAR(100),     -- 冗余存个名字
                    original_path NVARCHAR(MAX),    -- 原图在硬盘上的绝对路径
                    mask_path NVARCHAR(MAX),        -- 分割结果在硬盘上的绝对路径
                    diagnosis_note NVARCHAR(MAX),   -- 医生的诊断备注
                    created_at DATETIME DEFAULT GETDATE()
                )
            END
        """
        try:
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            print(f"❌ 初始化表失败: {e}")
        finally:
            conn.close()

    def save_record(self, p_code, p_name, current_img_path, mask_save_path, note=""):
        """
        核心功能：归档保存
        """
        # === 1. 【核心修复】去空格！解决 WinError 3 和路径错误 ===
        # 很多时候 Excel 或 DICOM 读出来的编号带空格（如 "P001 "），这会导致文件夹创建失败
        p_code = str(p_code).strip() 
        p_name = str(p_name).strip()
        # ======================================================

        conn = self.get_connection()
        if not conn: return False
        
        try:
            # 2. 规划存储路径 (使用绝对路径，防止找不到文件夹)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            save_root = os.path.join(current_file_dir, "saved_records")
            
            # 打印调试信息
            print(f"📂 存储根目录: {save_root}")
            print(f"👤 正在归档病人: [{p_code}] {p_name}") 
            
            patient_dir = os.path.join(save_root, p_code)
            if not os.path.exists(patient_dir):
                os.makedirs(patient_dir)
            
            # 3. 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # === 4. 复制原始影像（智能判断是文件夹还是文件） ===
            is_folder = os.path.isdir(current_img_path)
            target_org_path = ""

            if is_folder:
                # A. 如果是 DICOM 文件夹 -> 使用 copytree 复制整个目录
                new_folder_name = f"{timestamp}_dicom_series"
                target_org_path = os.path.join(patient_dir, new_folder_name)
                
                # 防止重名冲突
                if os.path.exists(target_org_path):
                    shutil.rmtree(target_org_path)
                    
                print(f"   📂 正在复制 DICOM 文件夹...")
                shutil.copytree(current_img_path, target_org_path)
                
            else:
                # B. 如果是单个文件 -> 使用 copy2 复制文件
                # === 【核心修复】智能后缀识别 ===
                # 防止把无后缀的 DICOM 文件 (如 IM5) 强行存成 .nii.gz
                src_dir, src_filename = os.path.split(current_img_path)
                filename_base, src_ext = os.path.splitext(src_filename)
                
                final_ext = ".nii.gz" # 默认值
                
                if src_ext == "" or src_ext.lower() == ".dcm":
                    final_ext = ".dcm" # 无后缀或 .dcm 都视为 DICOM
                elif src_ext.lower() == ".nii":
                    final_ext = ".nii"
                elif src_ext.lower() == ".gz":
                    final_ext = ".nii.gz"
                else:
                    final_ext = src_ext # 其他图片保留原后缀 (.jpg, .png)
                
                new_filename = f"{timestamp}_org{final_ext}"
                target_org_path = os.path.join(patient_dir, new_filename)
                
                print(f"   📄 正在复制文件 (后缀识别为 {final_ext})...")
                shutil.copy2(current_img_path, target_org_path)

            # === 5. 复制 Mask (结果图) ===
            new_mask_name = f"{timestamp}_mask.nii.gz"
            target_mask_path = os.path.join(patient_dir, new_mask_name)
            shutil.copy2(mask_save_path, target_mask_path)
            
            # === 6. 写入数据库 ===
            cursor = conn.cursor()
            sql = """
                INSERT INTO segmentation_records 
                (patient_code, patient_name, original_path, mask_path, diagnosis_note)
                VALUES (?, ?, ?, ?, ?)
            """
            cursor.execute(sql, (p_code, p_name, target_org_path, target_mask_path, note))
            conn.commit()
            print(f"✅ [Success] 归档成功！")
            return True
            
        except Exception as e:
            print(f"❌ [Error] 归档失败: {e}")
            traceback.print_exc() # 打印详细报错，方便排查
            return False
        finally:
            conn.close()

    def get_patient_history(self, keyword):
        """
        搜索功能：查找病人的所有历史记录
        """
        conn = self.get_connection()
        if not conn: return []
        
        try:
            cursor = conn.cursor()
            # 模糊搜索：名字或编号包含关键词
            sql = """
                SELECT id, patient_code, patient_name, created_at, original_path, mask_path, diagnosis_note
                FROM segmentation_records
                WHERE patient_name LIKE ? OR patient_code LIKE ?
                ORDER BY created_at DESC
            """
            search_term = f"%{keyword}%"
            cursor.execute(sql, (search_term, search_term))
            
            if cursor.description:
                columns = [column[0] for column in cursor.description]
                results = []
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return results
            else:
                return []
                
        except Exception as e:
            print(f"❌ 查询历史记录失败: {e}")
            return []
        finally:
            conn.close()