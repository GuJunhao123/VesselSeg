# case_data_service.py
from database import DatabaseManager

class CaseDataService:
    def __init__(self):
        self.db = DatabaseManager()

    def get_patient_excel_records(self, p_code):
        """
        从数据库的两个 Excel 对应表中查询数据
        假设数据库表名为: StatTable (数据统计) 和 CheckTable (检查检验)
        """
        conn = self.db.get_connection()
        if not conn:
            return None, None
            
        p_code = str(p_code).strip()
        stat_data = {}
        check_data = {}

        try:
            cursor = conn.cursor()

            # 1. 查询数据统计表 (根据编号匹配)
            # 请将 StatTable 替换为你实际的数据库表名
            sql_stat = "SELECT * FROM [dbo].[patient_info] WHERE 编号 = ?" 
            cursor.execute(sql_stat, (p_code,))
            row = cursor.fetchone()
            if row:
                # 将 pyodbc row 转为字典
                columns = [column[0] for column in cursor.description]
                stat_data = dict(zip(columns, row))

            # 2. 查询检查检验表
            # 请将 CheckTable 替换为你实际的数据库表名
            sql_check = "SELECT * FROM [dbo].[clinical_indicators] WHERE 病人编号 = ?"
            cursor.execute(sql_check, (p_code,))
            row = cursor.fetchone()
            if row:
                columns = [column[0] for column in cursor.description]
                check_data = dict(zip(columns, row))

        except Exception as e:
            print(f"❌ 数据库 Excel 项查询失败: {e}")
        finally:
            conn.close()

        return stat_data, check_data