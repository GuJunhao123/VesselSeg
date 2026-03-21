import pandas as pd
import os

class CaseDataManager:
    def __init__(self):
        # 预加载 Excel 文件以提高性能
        self.file_stat = "数据统计712.xlsx"
        self.file_check = "检查检验指标.xlsx"
        self.df_stat = None
        self.df_check = None
        self._load_all_excels()

    def _load_all_excels(self):
        """初始化时加载 Excel，避免重复读盘"""
        try:
            if os.path.exists(self.file_stat):
                self.df_stat = pd.read_excel(self.file_stat)
            if os.path.exists(self.file_check):
                self.df_check = pd.read_excel(self.file_check)
            print("✅ Excel 数据库预加载成功")
        except Exception as e:
            print(f"❌ 预加载 Excel 失败: {e}")

    def get_patient_data(self, p_code):
        """
        根据编号获取两个表的数据内容
        返回: (统计数据字典, 检验指标字典)
        """
        p_code_str = str(p_code).strip() # 解决空格导致的匹配失败
        stat_data = {}
        check_data = {}

        # 1. 匹配数据统计表 (假设列名为 '编号')
        if self.df_stat is not None:
            # 这里的匹配逻辑与你之前的要求一致
            res = self.df_stat[self.df_stat['编号'].astype(str).str.strip() == p_code_str]
            if not res.empty:
                stat_data = res.iloc[0].to_dict()

        # 2. 匹配检查检验表 (假设列名为 '病人编号')
        if self.df_check is not None:
            res = self.df_check[self.df_check['病人编号'].astype(str).str.strip() == p_code_str]
            if not res.empty:
                check_data = res.iloc[0].to_dict()

        return stat_data, check_data