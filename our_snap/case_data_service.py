# case_data_service.py
from database import DatabaseManager
import json
from config import Config

class CaseDataService:
    def __init__(self):
        self.db = DatabaseManager()

    def get_patient_excel_records(self, p_code):
        """
        从日志文件中查询患者记录
        """
        p_code = str(p_code).strip()
        stat_data = {}
        check_data = {}

        try:
            # 从日志文件中获取患者记录
            records = self.db.get_patient_history(p_code)
            
            if records:
                # 取最新的记录
                latest_record = records[0]
                stat_data = {
                    "编号": latest_record.get('patient_code', ''),
                    "姓名": latest_record.get('patient_name', ''),
                    "分割时间": latest_record.get('created_at', ''),
                    "备注": latest_record.get('diagnosis_note', '')
                }
                
                check_data = {
                    "病人编号": latest_record.get('patient_code', ''),
                    "检查时间": latest_record.get('created_at', ''),
                    "影像路径": latest_record.get('original_path', ''),
                    "分割结果路径": latest_record.get('mask_path', '')
                }

        except Exception as e:
            print(f"❌ 日志查询失败: {e}")
        
        return stat_data, check_data
