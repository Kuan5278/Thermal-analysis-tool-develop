# thermal_analysis_platform_v10.4.0_with_burn_in_test.py
# 溫度數據視覺化平台 - v10.4.0 新增燒機測試支援版

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime, date
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

# 版本資訊
VERSION = "v10.4.0 with Burn-In Test Support"
VERSION_DATE = "2025年6月"

# =============================================================================
# 0. 訪問計數器 (Visit Counter)
# =============================================================================

class VisitCounter:
    """訪問計數器"""
    
    def __init__(self, counter_file="visit_counter.json"):
        self.counter_file = counter_file
        self.data = self._load_counter()
    
    def _load_counter(self) -> dict:
        """載入計數器數據"""
        try:
            if os.path.exists(self.counter_file):
                with open(self.counter_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {
                    "total_visits": 0,
                    "daily_visits": {},
                    "first_visit": None,
                    "last_visit": None
                }
        except Exception:
            return {
                "total_visits": 0,
                "daily_visits": {},
                "first_visit": None,
                "last_visit": None
            }
    
    def _save_counter(self):
        """保存計數器數據"""
        try:
            with open(self.counter_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def increment_visit(self):
        """增加訪問計數"""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        # 更新總訪問次數
        self.data["total_visits"] += 1
        
        # 更新今日訪問次數
        if today not in self.data["daily_visits"]:
            self.data["daily_visits"][today] = 0
        self.data["daily_visits"][today] += 1
        
        # 更新首次訪問時間
        if self.data["first_visit"] is None:
            self.data["first_visit"] = now.isoformat()
        
        # 更新最後訪問時間
        self.data["last_visit"] = now.isoformat()
        
        # 清理舊的日訪問記錄（保留最近30天）
        self._cleanup_old_records()
        
        # 保存數據
        self._save_counter()
    
    def _cleanup_old_records(self):
        """清理30天前的日訪問記錄"""
        try:
            today = date.today()
            cutoff_date = today.replace(day=today.day-30) if today.day > 30 else today.replace(month=today.month-1, day=30)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            
            # 移除30天前的記錄
            keys_to_remove = [k for k in self.data["daily_visits"].keys() if k < cutoff_str]
            for key in keys_to_remove:
                del self.data["daily_visits"][key]
        except Exception:
            pass
    
    def get_stats(self) -> dict:
        """獲取統計信息"""
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today().replace(day=date.today().day-1)).strftime("%Y-%m-%d") if date.today().day > 1 else None
        
        # 計算最近7天訪問量
        recent_7_days = 0
        for i in range(7):
            check_date = (date.today().replace(day=date.today().day-i)).strftime("%Y-%m-%d")
            recent_7_days += self.data["daily_visits"].get(check_date, 0)
        
        return {
            "total_visits": self.data["total_visits"],
            "today_visits": self.data["daily_visits"].get(today, 0),
            "yesterday_visits": self.data["daily_visits"].get(yesterday, 0) if yesterday else 0,
            "recent_7_days": recent_7_days,
            "first_visit": self.data["first_visit"],
            "last_visit": self.data["last_visit"],
            "active_days": len(self.data["daily_visits"])
        }

def display_visit_counter():
    """顯示訪問計數器"""
    # 初始化計數器
    if 'visit_counter' not in st.session_state:
        st.session_state.visit_counter = VisitCounter()
        st.session_state.visit_counted = False
    
    # 只在第一次加載時計數
    if not st.session_state.visit_counted:
        st.session_state.visit_counter.increment_visit()
        st.session_state.visit_counted = True
    
    # 獲取統計數據
    stats = st.session_state.visit_counter.get_stats()
    
    # 顯示計數器
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 使用統計")
        
        # 使用columns來並排顯示
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="💫 總訪問",
                value=f"{stats['total_visits']:,}",
                help="自首次啟動以來的總訪問次數"
            )
            
            st.metric(
                label="📅 今日",
                value=f"{stats['today_visits']:,}",
                delta=f"+{stats['today_visits'] - stats['yesterday_visits']}" if stats['yesterday_visits'] > 0 else None,
                help="今日訪問次數"
            )
        
        with col2:
            st.metric(
                label="📈 近7天",
                value=f"{stats['recent_7_days']:,}",
                help="最近7天總訪問次數"
            )
            
            st.metric(
                label="🗓️ 活躍天數",
                value=f"{stats['active_days']:,}",
                help="有訪問記錄的天數"
            )

# =============================================================================
# 1. 數據模型層 (Data Model Layer)
# =============================================================================

@dataclass
class LogMetadata:
    """Log檔案元數據"""
    filename: str
    log_type: str
    rows: int
    columns: int
    time_range: str
    file_size_kb: float

class LogData:
    """統一的Log數據抽象類"""
    def __init__(self, df: pd.DataFrame, metadata: LogMetadata):
        self.df = df
        self.metadata = metadata
        self._numeric_columns = None
    
    @property
    def numeric_columns(self) -> List[str]:
        """獲取數值型欄位"""
        if self._numeric_columns is None:
            self._numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        return self._numeric_columns
    
    def get_time_range(self) -> Tuple[float, float]:
        """獲取時間範圍（秒）"""
        if self.df.empty:
            return (0.0, 0.0)
        return (0.0, self.df.index.total_seconds().max())
    
    def filter_by_time(self, x_limits: Tuple[float, float]):
        """按時間範圍過濾數據"""
        if x_limits is None:
            return self.df
        
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        return self.df[(self.df.index >= x_min_td) & (self.df.index <= x_max_td)]

# =============================================================================
# 2. 解析器層 (Parser Layer)
# =============================================================================

class ParseLogger:
    """解析日誌管理器"""
    
    def __init__(self):
        self.logs = []
        self.debug_logs = []
        self.success_logs = []
        self.error_logs = []
    
    def info(self, message: str):
        """記錄一般信息"""
        self.logs.append(f"ℹ️ {message}")
    
    def debug(self, message: str):
        """記錄調試信息"""
        self.debug_logs.append(f"🔍 {message}")
    
    def success(self, message: str):
        """記錄成功信息"""
        self.success_logs.append(f"✅ {message}")
    
    def error(self, message: str):
        """記錄錯誤信息"""
        self.error_logs.append(f"❌ {message}")
    
    def warning(self, message: str):
        """記錄警告信息"""
        self.logs.append(f"⚠️ {message}")
    
    def show_summary(self, filename: str, log_type: str):
        """顯示簡潔的解析摘要"""
        if self.success_logs:
            st.success(f"✅ {log_type} 解析成功！")
        elif self.error_logs:
            st.error(f"❌ {filename} 解析失敗")
            return
    
    def show_detailed_logs(self, filename: str):
        """在摺疊區域內顯示詳細日誌"""
        with st.expander(f"🔍 詳細解析日誌 - {filename}", expanded=False):
            if self.debug_logs:
                st.markdown("**🔍 調試信息：**")
                for log in self.debug_logs:
                    st.code(log, language=None)
            
            if self.logs:
                st.markdown("**📋 解析過程：**")
                for log in self.logs:
                    st.write(log)
            
            if self.success_logs:
                st.markdown("**✅ 成功信息：**")
                for log in self.success_logs:
                    st.write(log)
            
            if self.error_logs:
                st.markdown("**❌ 錯誤信息：**")
                for log in self.error_logs:
                    st.write(log)

class LogParser(ABC):
    """解析器抽象基類"""
    
    def __init__(self):
        self.logger = ParseLogger()
    
    @abstractmethod
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        """判斷是否能解析此檔案"""
        pass
    
    @abstractmethod
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        """解析檔案"""
        pass
    
    @property
    @abstractmethod
    def log_type(self) -> str:
        """Log類型名稱"""
        pass

# =============================================================================
# 新增：燒機測試解析器 (Burn-In Test Parser)
# =============================================================================

class BurnInTestParser(LogParser):
    """燒機測試Log解析器"""
    
    @property
    def log_type(self) -> str:
        return "Burn-In Test Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        """檢查是否為燒機測試格式"""
        try:
            file_content.seek(0)
            content = file_content.read(2000).decode('utf-8', errors='ignore')
            
            # 檢查燒機測試特有的格式標誌
            indicators = [
                'Start Burn In Test' in content,
                'cpu0 freq:' in content and 'cpu0 temp:' in content,
                'gpu temp:' in content,
                'Board temperature:' in content,
                'USB Disk Test' in content and 'ttyHS1 Test' in content,
                # 時間戳格式檢查
                bool(re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', content))
            ]
            
            # 至少要符合3個條件
            return sum(indicators) >= 3
            
        except Exception as e:
            self.logger.debug(f"格式檢查異常: {e}")
            return False
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        """解析燒機測試檔案"""
        try:
            file_content.seek(0)
            content = file_content.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            
            self.logger.debug(f"檔案總行數: {len(lines)}")
            
            # 解析數據
            parsed_records = self._parse_burn_in_data(lines)
            if not parsed_records:
                self.logger.error("沒有找到有效的燒機測試數據")
                return None
            
            # 轉換為DataFrame
            df = self._create_dataframe(parsed_records)
            if df is None or df.empty:
                self.logger.error("DataFrame創建失敗")
                return None
            
            # 添加前綴並設置索引
            df = df.add_prefix('BURN: ')
            df.rename(columns={'BURN: time_index': 'time_index'}, inplace=True)
            result_df = df.set_index('time_index')
            
            # 創建元數據
            file_size_kb = len(content.encode('utf-8')) / 1024
            time_range = f"{result_df.index.min()} 到 {result_df.index.max()}"
            
            metadata = LogMetadata(
                filename=filename,
                log_type=self.log_type,
                rows=result_df.shape[0],
                columns=result_df.shape[1],
                time_range=time_range,
                file_size_kb=file_size_kb
            )
            
            self.logger.success(f"燒機測試解析成功！數據形狀: {result_df.shape}")
            return LogData(result_df, metadata)
            
        except Exception as e:
            self.logger.error(f"燒機測試解析異常: {e}")
            return None
    
    def _parse_burn_in_data(self, lines: List[str]) -> List[Dict]:
        """解析燒機測試數據"""
        records = []
        current_record = {}
        current_timestamp = None
        
        self.logger.debug("開始解析燒機測試數據...")
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析時間戳
                timestamp_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    # 保存上一條記錄
                    if current_timestamp and current_record:
                        current_record['timestamp'] = current_timestamp
                        records.append(current_record.copy())
                    
                    # 開始新記錄
                    current_timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    current_record = {}
                    continue
                
                # 解析CPU頻率
                cpu_freq_match = re.match(r'^cpu(\d+) freq:(\d+)
                
                # 解析CPU溫度
                cpu_temp_match = re.match(r'^cpu(\d+) temp:(\d+)$', line)
                if cpu_temp_match:
                    cpu_index = int(cpu_temp_match.group(1))
                    temp_millidegree = int(cpu_temp_match.group(2))
                    temp_celsius = temp_millidegree / 1000  # 轉換為攝氏度
                    current_record[f'CPU{cpu_index}_Temp_C'] = temp_celsius
                    continue
                
                # 解析GPU溫度
                gpu_temp_match = re.match(r'^gpu temp:(\d+)$', line)
                if gpu_temp_match:
                    temp_millidegree = int(gpu_temp_match.group(1))
                    temp_celsius = temp_millidegree / 1000
                    if 'GPU_Temp_C' not in current_record:
                        current_record['GPU_Temp_C'] = []
                    current_record['GPU_Temp_C'].append(temp_celsius)
                    continue
                
                # 解析主板溫度
                board_temp_match = re.match(r'^Board temperature: ([\d.]+)$', line)
                if board_temp_match:
                    temp_celsius = float(board_temp_match.group(1))
                    current_record['Board_Temp_C'] = temp_celsius
                    continue
                
            except Exception as e:
                self.logger.debug(f"第{line_num+1}行解析異常: {e}")
                continue
        
        # 保存最後一條記錄
        if current_timestamp and current_record:
            current_record['timestamp'] = current_timestamp
            records.append(current_record)
        
        self.logger.debug(f"成功解析 {len(records)} 條記錄")
        return records
    
    def _create_dataframe(self, records: List[Dict]) -> Optional[pd.DataFrame]:
        """創建DataFrame"""
        if not records:
            return None
        
        try:
            # 處理GPU溫度（可能有多個值，取平均）
            for record in records:
                if 'GPU_Temp_C' in record and isinstance(record['GPU_Temp_C'], list):
                    if record['GPU_Temp_C']:
                        record['GPU_Temp_C'] = sum(record['GPU_Temp_C']) / len(record['GPU_Temp_C'])
                    else:
                        record['GPU_Temp_C'] = None
            
            # 創建DataFrame
            df = pd.DataFrame(records)
            
            if 'timestamp' not in df.columns:
                self.logger.error("缺少時間戳信息")
                return None
            
            # 計算時間差
            start_time = df['timestamp'].iloc[0]
            df['time_index'] = df['timestamp'] - start_time
            
            # 移除原始時間戳列
            df = df.drop(['timestamp'], axis=1)
            
            # 填充缺失值
            df = df.fillna(method='ffill').fillna(0)
            
            self.logger.debug(f"DataFrame創建成功: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"DataFrame創建失敗: {e}")
            return None

# 其他解析器保持不變...
class GPUMonParser(LogParser):
    """GPUMon解析器"""
    
    @property
    def log_type(self) -> str:
        return "GPUMon Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            first_content = ""
            for _ in range(100):
                try:
                    line = file_content.readline().decode('utf-8', errors='ignore')
                    if not line:
                        break
                    first_content += line
                except:
                    break
            
            indicators = [
                'GPU Informations' in first_content,
                'Iteration, Date, Timestamp' in first_content,
                'Temperature GPU (C)' in first_content,
                'iteration' in first_content.lower() and 'gpu' in first_content.lower(),
                'NVVDD' in first_content,
                'FBVDD' in first_content
            ]
            
            return any(indicators)
        except:
            return False
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        # GPUMon解析邏輯保持不變
        return None  # 簡化示例

class PTATParser(LogParser):
    """PTAT解析器"""
    
    @property
    def log_type(self) -> str:
        return "PTAT Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            first_content = file_content.read(2000).decode('utf-8', errors='ignore')
            return ('MSR Package Temperature' in first_content or 
                    'Version,Date,Time' in first_content)
        except:
            return False
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        # PTAT解析邏輯保持不變
        return None  # 簡化示例

class YokogawaParser(LogParser):
    """YOKOGAWA解析器"""
    
    @property
    def log_type(self) -> str:
        return "YOKOGAWA Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        return True  # 兜底解析器
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        # YOKOGAWA解析邏輯保持不變
        return None  # 簡化示例

# =============================================================================
# 3. 解析器註冊系統 (Parser Registry)
# =============================================================================

class ParserRegistry:
    """解析器註冊系統"""
    
    def __init__(self):
        self.parsers: List[LogParser] = []
    
    def register(self, parser: LogParser):
        """註冊解析器"""
        self.parsers.append(parser)
    
    def parse_file(self, uploaded_file) -> Optional[LogData]:
        """解析檔案，自動選擇合適的解析器"""
        filename = uploaded_file.name
        file_content = io.BytesIO(uploaded_file.getvalue())
        
        for parser in self.parsers:
            try:
                file_content.seek(0)
                if parser.can_parse(file_content, filename):
                    file_content.seek(0)
                    result = parser.parse(file_content, filename)
                    if result is not None:
                        parser.logger.show_summary(filename, parser.log_type)
                        parser.logger.show_detailed_logs(filename)
                        return result
            except Exception as e:
                continue
        
        st.error(f"❌ 無法解析檔案 {filename}")
        return None

# =============================================================================
# 4. 統計計算層 (Statistics Layer)
# =============================================================================

class StatisticsCalculator:
    """統計計算器"""
    
    @staticmethod
    def calculate_burn_in_stats(log_data: LogData, x_limits=None):
        """計算燒機測試統計數據"""
        df = log_data.filter_by_time(x_limits)
        if df.empty:
            return None, None
        
        # CPU溫度統計
        temp_stats = []
        cpu_temp_cols = [col for col in df.columns if 'CPU' in col and 'Temp' in col]
        
        for col in sorted(cpu_temp_cols):
            temp_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(temp_data) > 0:
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                temp_stats.append({
                    'CPU Core': core_name,
                    'Max Temp (°C)': f"{temp_data.max():.1f}",
                    'Min Temp (°C)': f"{temp_data.min():.1f}",
                    'Avg Temp (°C)': f"{temp_data.mean():.1f}"
                })
        
        temp_df = pd.DataFrame(temp_stats) if temp_stats else None
        
        # CPU頻率統計
        freq_stats = []
        cpu_freq_cols = [col for col in df.columns if 'CPU' in col and 'Freq' in col]
        
        for col in sorted(cpu_freq_cols):
            freq_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(freq_data) > 0:
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                freq_stats.append({
                    'CPU Core': core_name,
                    'Max Freq (kHz)': f"{freq_data.max():.0f}",
                    'Min Freq (kHz)': f"{freq_data.min():.0f}",
                    'Avg Freq (kHz)': f"{freq_data.mean():.0f}"
                })
        
        freq_df = pd.DataFrame(freq_stats) if freq_stats else None
        
        return temp_df, freq_df

# =============================================================================
# 5. 圖表生成層 (Chart Generation Layer)
# =============================================================================

class ChartGenerator:
    """圖表生成器"""
    
    @staticmethod
    def generate_burn_in_temp_chart(log_data: LogData, x_limits=None, y_limits=None):
        """生成燒機測試CPU溫度圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 獲取所有CPU溫度欄位
        temp_cols = [col for col in df.columns if 'CPU' in col and 'Temp' in col]
        temp_cols = sorted(temp_cols)
        
        # 定義顏色
        colors = plt.cm.tab10(np.linspace(0, 1, len(temp_cols)))
        
        for i, col in enumerate(temp_cols):
            temp_data = pd.to_numeric(df[col], errors='coerce')
            if not temp_data.isna().all():
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                ax.plot(df.index.total_seconds(), temp_data, 
                       label=core_name, color=colors[i], linewidth=2)
        
        ax.set_title("🔥 燒機測試 - CPU各核心溫度變化", fontsize=14, fontweight='bold')
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="CPU Cores", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
        
        if x_limits:
            ax.set_xlim(x_limits)
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def generate_burn_in_freq_chart(log_data: LogData, x_limits=None, y_limits=None):
        """生成燒機測試CPU頻率圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 獲取所有CPU頻率欄位
        freq_cols = [col for col in df.columns if 'CPU' in col and 'Freq' in col]
        freq_cols = sorted(freq_cols)
        
        # 定義顏色
        colors = plt.cm.Set1(np.linspace(0, 1, len(freq_cols)))
        
        for i, col in enumerate(freq_cols):
            freq_data = pd.to_numeric(df[col], errors='coerce')
            if not freq_data.isna().all():
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                ax.plot(df.index.total_seconds(), freq_data, 
                       label=core_name, color=colors[i], linewidth=2)
        
        ax.set_title("⚡ 燒機測試 - CPU各核心頻率變化", fontsize=14, fontweight='bold')
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Frequency (kHz)", fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="CPU Cores", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
        
        if x_limits:
            ax.set_xlim(x_limits)
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        return fig

# =============================================================================
# 6. 燒機測試比較功能 (Burn-In Test Comparison)
# =============================================================================

class BurnInTestComparison:
    """燒機測試比較器"""
    
    @staticmethod
    def generate_comparison_table(log_data_list: List[LogData]) -> pd.DataFrame:
        """生成燒機測試比較表格"""
        burn_in_logs = [log for log in log_data_list if log.metadata.log_type == "Burn-In Test Log"]
        
        if len(burn_in_logs) < 2:
            return pd.DataFrame()
        
        comparison_data = []
        
        # 獲取所有CPU核心
        all_cores = set()
        for log_data in burn_in_logs:
            df = log_data.df
            temp_cols = [col for col in df.columns if 'CPU' in col and 'Temp' in col]
            for col in temp_cols:
                core_name = col.replace('BURN: ', '').replace('_Temp_C', '')
                all_cores.add(core_name)
        
        all_cores = sorted(all_cores)
        
        for core in all_cores:
            row_data = {'CPU Core': core}
            
            for i, log_data in enumerate(burn_in_logs):
                df = log_data.df
                test_name = f"Test {i+1} ({log_data.metadata.filename})"
                
                # 找對應的溫度欄位
                temp_col = f'BURN: {core}_Temp_C'
                if temp_col in df.columns:
                    temp_data = pd.to_numeric(df[temp_col], errors='coerce').dropna()
                    if len(temp_data) > 0:
                        max_temp = temp_data.max()
                        avg_temp = temp_data.mean()
                        row_data[f'{test_name} Max (°C)'] = f"{max_temp:.1f}"
                        row_data[f'{test_name} Avg (°C)'] = f"{avg_temp:.1f}"
                    else:
                        row_data[f'{test_name} Max (°C)'] = "N/A"
                        row_data[f'{test_name} Avg (°C)'] = "N/A"
                else:
                    row_data[f'{test_name} Max (°C)'] = "N/A"
                    row_data[f'{test_name} Avg (°C)'] = "N/A"
            
            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)

# =============================================================================
# 7. UI渲染層 (UI Rendering Layer)
# =============================================================================

class BurnInTestRenderer:
    """燒機測試UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render_controls(self, file_index=None):
        """渲染控制面板"""
        if file_index is None:
            file_index = getattr(st.session_state, 'current_file_index', 0)
        key_prefix = f"burn_{file_index}_"
        
        st.sidebar.markdown("### ⚙️ 燒機測試圖表設定")
        
        st.sidebar.markdown("#### 📈 圖表類型選擇")
        chart_type = st.sidebar.radio(
            "選擇圖表類型", 
            ["CPU溫度圖", "CPU頻率圖"],
            key=f"{key_prefix}chart_type"
        )
        
        st.sidebar.markdown("#### ⏱️ 時間範圍設定")
        time_min, time_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider(
            "選擇時間範圍 (秒)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
            step=1.0,
            key=f"{key_prefix}x_range"
        )
        
        st.sidebar.markdown("#### 📏 Y軸範圍設定")
        y_range_enabled = st.sidebar.checkbox("啟用Y軸範圍限制", key=f"{key_prefix}y_range_enabled")
        y_range = None
        if y_range_enabled:
            if chart_type == "CPU溫度圖":
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    y_min = st.number_input("Y軸最小值 (°C)", value=30.0, step=1.0, key=f"{key_prefix}y_min")
                with col2:
                    y_max = st.number_input("Y軸最大值 (°C)", value=100.0, step=1.0, key=f"{key_prefix}y_max")
            else:  # CPU頻率圖
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    y_min = st.number_input("Y軸最小值 (kHz)", value=0.0, step=100.0, key=f"{key_prefix}y_min")
                with col2:
                    y_max = st.number_input("Y軸最大值 (kHz)", value=3000000.0, step=100000.0, key=f"{key_prefix}y_max")
            
            y_range = (y_min, y_max)
        
        return chart_type, x_range, y_range
    
    def render_chart(self, chart_type, x_range, y_range):
        """渲染圖表"""
        st.markdown("### 📊 燒機測試監控圖表")
        
        if chart_type == "CPU溫度圖":
            chart = self.chart_gen.generate_burn_in_temp_chart(self.log_data, x_range, y_range)
        else:  # CPU頻率圖
            chart = self.chart_gen.generate_burn_in_freq_chart(self.log_data, x_range, y_range)
        
        if chart:
            st.pyplot(chart)
        else:
            st.warning("無法生成圖表，請檢查數據")
    
    def render_statistics(self, x_range):
        """渲染統計數據"""
        st.markdown("### 📈 燒機測試統計數據")
        
        temp_stats, freq_stats = self.stats_calc.calculate_burn_in_stats(self.log_data, x_range)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if temp_stats is not None and not temp_stats.empty:
                st.markdown("#### 🌡️ CPU溫度統計")
                st.dataframe(temp_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("無CPU溫度數據")
        
        with col2:
            if freq_stats is not None and not freq_stats.empty:
                st.markdown("#### ⚡ CPU頻率統計")
                st.dataframe(freq_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("無CPU頻率數據")
    
    def render(self, file_index=None):
        """渲染完整UI"""
        st.markdown("""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4>🔥 燒機測試Log 解析完成！</h4>
            <p>已識別為燒機測試數據，包含CPU溫度、頻率、GPU溫度等監控指標</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 顯示測試概況
        df = self.log_data.df
        cpu_count = len([col for col in df.columns if 'CPU' in col and 'Temp' in col])
        duration_minutes = self.log_data.df.index.total_seconds().max() / 60
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU核心數", f"{cpu_count}", help="監控的CPU核心數量")
        with col2:
            st.metric("測試時長", f"{duration_minutes:.1f} 分鐘", help="燒機測試總時長")
        with col3:
            st.metric("數據點數", f"{self.log_data.metadata.rows}", help="總記錄數據點")
        
        chart_type, x_range, y_range = self.render_controls(file_index)
        self.render_chart(chart_type, x_range, y_range)
        self.render_statistics(x_range)

# =============================================================================
# 8. UI工廠更新 (UI Factory Update)
# =============================================================================

class RendererFactory:
    """UI渲染器工廠"""
    
    @staticmethod
    def create_renderer(log_data: LogData):
        """根據log類型創建對應的渲染器"""
        log_type = log_data.metadata.log_type
        
        if log_type == "Burn-In Test Log":
            return BurnInTestRenderer(log_data)
        elif log_type == "GPUMon Log":
            # return GPUMonRenderer(log_data)  # 簡化示例
            return None
        elif log_type == "PTAT Log":
            # return PTATRenderer(log_data)  # 簡化示例
            return None
        elif log_type == "YOKOGAWA Log":
            # return YokogawaRenderer(log_data)  # 簡化示例
            return None
        else:
            return None

# =============================================================================
# 9. 主應用程式 (Main Application)
# =============================================================================

def display_version_info():
    """顯示版本資訊"""
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"""
        **當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**
        
        ### ✨ 主要功能
        
        - **🔥 Burn-In Test Log** - 燒機測試數據解析與視覺化 (新增)
        - **🎮 GPUMon Log** - GPU性能監控數據解析與視覺化
        - **🖥️ PTAT Log** - CPU性能監控數據解析與視覺化  
        - **📊 YOKOGAWA Log** - 多通道溫度記錄儀數據解析與視覺化
        - **📋 多檔案比較** - 燒機測試結果自動比較分析
        
        ### 🔥 燒機測試功能特色
        
        - **🌡️ CPU溫度監控** - 每個核心的溫度變化曲線
        - **⚡ CPU頻率監控** - 每個核心的頻率變化曲線 (單位：kHz)
        - **📊 統計分析** - 最大、最小、平均溫度/頻率
        - **🔄 多檔案比較** - 自動整合多次燒機測試結果
        - **⏱️ 時間範圍選擇** - 可自定義分析時間區間
        - **📏 Y軸自由調整** - 完全開放Y軸範圍設定
        """)

def main():
    """主程式 - v10.4.0 with Burn-In Test Support"""
    st.set_page_config(
        page_title="溫度數據視覺化平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS樣式
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .burn-in-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .comparison-table {
            font-size: 0.9em;
        }
        .comparison-table th {
            background-color: #f0f2f6;
            font-weight: bold;
            text-align: center;
        }
        .comparison-table td {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 標題
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 溫度數據視覺化平台</h1>
        <p>智能解析 YOKOGAWA、PTAT、GPUMon、燒機測試 Log 文件</p>
        <p><strong>{VERSION}</strong> | {VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    display_version_info()
    
    # 初始化解析器註冊系統
    parser_registry = ParserRegistry()
    parser_registry.register(BurnInTestParser())  # 新增燒機測試解析器
    parser_registry.register(GPUMonParser())
    parser_registry.register(PTATParser())
    parser_registry.register(YokogawaParser())  # 兜底解析器
    
    # 側邊欄
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx', 'log', 'txt'], 
        accept_multiple_files=True,
        help="v10.4.0 新增燒機測試支援，支援多檔案比較分析"
    )
    
    # 顯示訪問計數器
    display_visit_counter()
    
    if uploaded_files:
        # 顯示上傳檔案資訊
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 已上傳檔案")
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / 1024
            st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
        
        st.sidebar.markdown("---")
        
        # 解析檔案
        log_data_list = []
        for uploaded_file in uploaded_files:
            log_data = parser_registry.parse_file(uploaded_file)
            if log_data:
                log_data_list.append(log_data)
        
        if not log_data_list:
            st.error("❌ 無法解析任何檔案")
            return
        
        # 檢查是否有燒機測試檔案
        burn_in_logs = [log for log in log_data_list if log.metadata.log_type == "Burn-In Test Log"]
        
        if len(burn_in_logs) > 1:
            # 多個燒機測試檔案 - 顯示比較功能
            st.success(f"🔥 燒機測試比較模式：找到 {len(burn_in_logs)} 個燒機測試檔案")
            
            # 創建標籤頁
            tab_names = ["📋 比較分析"] + [f"🔥 測試 {i+1}" for i in range(len(burn_in_logs))]
            tabs = st.tabs(tab_names)
            
            # 比較分析標籤頁
            with tabs[0]:
                st.markdown("### 🔄 燒機測試比較分析")
                
                comparison_df = BurnInTestComparison.generate_comparison_table(burn_in_logs)
                if not comparison_df.empty:
                    st.markdown("#### 🌡️ CPU溫度比較表格")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # 顯示比較摘要
                    st.markdown("#### 📊 比較摘要")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("測試數量", f"{len(burn_in_logs)}", help="比較的燒機測試數量")
                    with col2:
                        total_duration = sum([log.df.index.total_seconds().max() for log in burn_in_logs]) / 60
                        st.metric("總測試時長", f"{total_duration:.1f} 分鐘", help="所有測試的總時長")
                    with col3:
                        cpu_cores = len([col for col in burn_in_logs[0].df.columns if 'CPU' in col and 'Temp' in col])
                        st.metric("監控CPU核心", f"{cpu_cores}", help="監控的CPU核心數量")
                else:
                    st.warning("無法生成比較表格")
            
            # 各個測試的詳細分析
            for i, (tab, log_data) in enumerate(zip(tabs[1:], burn_in_logs)):
                with tab:
                    renderer = BurnInTestRenderer(log_data)
                    renderer.render(file_index=i)
        
        elif len(log_data_list) == 1:
            # 單檔案模式
            log_data = log_data_list[0]
            renderer = RendererFactory.create_renderer(log_data)
            
            if renderer:
                renderer.render(file_index=0)
            else:
                st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
        
        else:
            # 多檔案混合模式
            st.success(f"📊 多檔案分析模式：成功解析 {len(log_data_list)} 個檔案")
            
            # 創建標籤頁
            tab_names = []
            for i, log_data in enumerate(log_data_list):
                filename = log_data.metadata.filename
                log_type = log_data.metadata.log_type
                
                short_name = filename[:12] + "..." if len(filename) > 15 else filename
                
                if "Burn-In Test" in log_type:
                    tab_name = f"🔥 {short_name}"
                elif "GPUMon" in log_type:
                    tab_name = f"🎮 {short_name}"
                elif "PTAT" in log_type:
                    tab_name = f"🖥️ {short_name}"
                elif "YOKOGAWA" in log_type:
                    tab_name = f"📊 {short_name}"
                else:
                    tab_name = f"📄 {short_name}"
                
                tab_names.append(tab_name)
            
            tabs = st.tabs(tab_names)
            
            for i, (tab, log_data) in enumerate(zip(tabs, log_data_list)):
                with tab:
                    # 顯示檔案資訊
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #1f77b4;">
                        <h4>📁 檔案資訊</h4>
                        <p><strong>檔案名稱：</strong> {log_data.metadata.filename}</p>
                        <p><strong>檔案類型：</strong> {log_data.metadata.log_type}</p>
                        <p><strong>數據規模：</strong> {log_data.metadata.rows} 行 × {log_data.metadata.columns} 列</p>
                        <p><strong>檔案大小：</strong> {log_data.metadata.file_size_kb:.1f} KB</p>
                        <p><strong>時間範圍：</strong> {log_data.metadata.time_range}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    renderer = RendererFactory.create_renderer(log_data)
                    
                    if renderer:
                        renderer.render(file_index=i)
                    else:
                        st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
    
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件進行分析")
        
        st.markdown("""
        ### 📋 支援的檔案格式
        
        - **🔥 燒機測試 Log** - CPU/GPU燒機測試監控數據 (新增支援)
        - **🎮 GPUMon CSV** - GPU性能監控數據（溫度、功耗、頻率、使用率）
        - **🖥️ PTAT CSV** - CPU性能監控數據（頻率、功耗、溫度）
        - **📊 YOKOGAWA Excel/CSV** - 多通道溫度記錄儀數據
        
        ### 🔥 燒機測試特色功能
        
        - **🌡️ CPU溫度圖表** - 每個核心的溫度變化曲線
        - **⚡ CPU頻率圖表** - 每個核心的頻率變化曲線 (單位：kHz)
        - **📊 統計分析** - 最大、最小、平均溫度和頻率
        - **🔄 多檔案比較** - 自動整合多次燒機測試結果
        - **⏱️ 時間範圍控制** - 可自定義分析的時間區間
        - **📏 Y軸自由設定** - 完全開放Y軸最小值和最大值調整
        """)

if __name__ == "__main__":
    main(), line)
                if cpu_freq_match:
                    cpu_index = int(cpu_freq_match.group(1))
                    freq_hz = int(cpu_freq_match.group(2))
                    freq_khz = freq_hz / 1000  # 轉換為kHz
                    current_record[f'CPU{cpu_index}_Freq_kHz'] = freq_khz
                    continue
                
                # 解析CPU溫度
                cpu_temp_match = re.match(r'^cpu(\d+) temp:(\d+)$', line)
                if cpu_temp_match:
                    cpu_index = int(cpu_temp_match.group(1))
                    temp_millidegree = int(cpu_temp_match.group(2))
                    temp_celsius = temp_millidegree / 1000  # 轉換為攝氏度
                    current_record[f'CPU{cpu_index}_Temp_C'] = temp_celsius
                    continue
                
                # 解析GPU溫度
                gpu_temp_match = re.match(r'^gpu temp:(\d+)$', line)
                if gpu_temp_match:
                    temp_millidegree = int(gpu_temp_match.group(1))
                    temp_celsius = temp_millidegree / 1000
                    if 'GPU_Temp_C' not in current_record:
                        current_record['GPU_Temp_C'] = []
                    current_record['GPU_Temp_C'].append(temp_celsius)
                    continue
                
                # 解析主板溫度
                board_temp_match = re.match(r'^Board temperature: ([\d.]+)$', line)
                if board_temp_match:
                    temp_celsius = float(board_temp_match.group(1))
                    current_record['Board_Temp_C'] = temp_celsius
                    continue
                
            except Exception as e:
                self.logger.debug(f"第{line_num+1}行解析異常: {e}")
                continue
        
        # 保存最後一條記錄
        if current_timestamp and current_record:
            current_record['timestamp'] = current_timestamp
            records.append(current_record)
        
        self.logger.debug(f"成功解析 {len(records)} 條記錄")
        return records
    
    def _create_dataframe(self, records: List[Dict]) -> Optional[pd.DataFrame]:
        """創建DataFrame"""
        if not records:
            return None
        
        try:
            # 處理GPU溫度（可能有多個值，取平均）
            for record in records:
                if 'GPU_Temp_C' in record and isinstance(record['GPU_Temp_C'], list):
                    if record['GPU_Temp_C']:
                        record['GPU_Temp_C'] = sum(record['GPU_Temp_C']) / len(record['GPU_Temp_C'])
                    else:
                        record['GPU_Temp_C'] = None
            
            # 創建DataFrame
            df = pd.DataFrame(records)
            
            if 'timestamp' not in df.columns:
                self.logger.error("缺少時間戳信息")
                return None
            
            # 計算時間差
            start_time = df['timestamp'].iloc[0]
            df['time_index'] = df['timestamp'] - start_time
            
            # 移除原始時間戳列
            df = df.drop(['timestamp'], axis=1)
            
            # 填充缺失值
            df = df.fillna(method='ffill').fillna(0)
            
            self.logger.debug(f"DataFrame創建成功: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"DataFrame創建失敗: {e}")
            return None

# 其他解析器保持不變...
class GPUMonParser(LogParser):
    """GPUMon解析器"""
    
    @property
    def log_type(self) -> str:
        return "GPUMon Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            first_content = ""
            for _ in range(100):
                try:
                    line = file_content.readline().decode('utf-8', errors='ignore')
                    if not line:
                        break
                    first_content += line
                except:
                    break
            
            indicators = [
                'GPU Informations' in first_content,
                'Iteration, Date, Timestamp' in first_content,
                'Temperature GPU (C)' in first_content,
                'iteration' in first_content.lower() and 'gpu' in first_content.lower(),
                'NVVDD' in first_content,
                'FBVDD' in first_content
            ]
            
            return any(indicators)
        except:
            return False
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        # GPUMon解析邏輯保持不變
        return None  # 簡化示例

class PTATParser(LogParser):
    """PTAT解析器"""
    
    @property
    def log_type(self) -> str:
        return "PTAT Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            first_content = file_content.read(2000).decode('utf-8', errors='ignore')
            return ('MSR Package Temperature' in first_content or 
                    'Version,Date,Time' in first_content)
        except:
            return False
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        # PTAT解析邏輯保持不變
        return None  # 簡化示例

class YokogawaParser(LogParser):
    """YOKOGAWA解析器"""
    
    @property
    def log_type(self) -> str:
        return "YOKOGAWA Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        return True  # 兜底解析器
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        # YOKOGAWA解析邏輯保持不變
        return None  # 簡化示例

# =============================================================================
# 3. 解析器註冊系統 (Parser Registry)
# =============================================================================

class ParserRegistry:
    """解析器註冊系統"""
    
    def __init__(self):
        self.parsers: List[LogParser] = []
    
    def register(self, parser: LogParser):
        """註冊解析器"""
        self.parsers.append(parser)
    
    def parse_file(self, uploaded_file) -> Optional[LogData]:
        """解析檔案，自動選擇合適的解析器"""
        filename = uploaded_file.name
        file_content = io.BytesIO(uploaded_file.getvalue())
        
        for parser in self.parsers:
            try:
                file_content.seek(0)
                if parser.can_parse(file_content, filename):
                    file_content.seek(0)
                    result = parser.parse(file_content, filename)
                    if result is not None:
                        parser.logger.show_summary(filename, parser.log_type)
                        parser.logger.show_detailed_logs(filename)
                        return result
            except Exception as e:
                continue
        
        st.error(f"❌ 無法解析檔案 {filename}")
        return None

# =============================================================================
# 4. 統計計算層 (Statistics Layer)
# =============================================================================

class StatisticsCalculator:
    """統計計算器"""
    
    @staticmethod
    def calculate_burn_in_stats(log_data: LogData, x_limits=None):
        """計算燒機測試統計數據"""
        df = log_data.filter_by_time(x_limits)
        if df.empty:
            return None, None
        
        # CPU溫度統計
        temp_stats = []
        cpu_temp_cols = [col for col in df.columns if 'CPU' in col and 'Temp' in col]
        
        for col in sorted(cpu_temp_cols):
            temp_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(temp_data) > 0:
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                temp_stats.append({
                    'CPU Core': core_name,
                    'Max Temp (°C)': f"{temp_data.max():.1f}",
                    'Min Temp (°C)': f"{temp_data.min():.1f}",
                    'Avg Temp (°C)': f"{temp_data.mean():.1f}"
                })
        
        temp_df = pd.DataFrame(temp_stats) if temp_stats else None
        
        # CPU頻率統計
        freq_stats = []
        cpu_freq_cols = [col for col in df.columns if 'CPU' in col and 'Freq' in col]
        
        for col in sorted(cpu_freq_cols):
            freq_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(freq_data) > 0:
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                freq_stats.append({
                    'CPU Core': core_name,
                    'Max Freq (MHz)': f"{freq_data.max():.0f}",
                    'Min Freq (MHz)': f"{freq_data.min():.0f}",
                    'Avg Freq (MHz)': f"{freq_data.mean():.0f}"
                })
        
        freq_df = pd.DataFrame(freq_stats) if freq_stats else None
        
        return temp_df, freq_df

# =============================================================================
# 5. 圖表生成層 (Chart Generation Layer)
# =============================================================================

class ChartGenerator:
    """圖表生成器"""
    
    @staticmethod
    def generate_burn_in_temp_chart(log_data: LogData, x_limits=None, y_limits=None):
        """生成燒機測試CPU溫度圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 獲取所有CPU溫度欄位
        temp_cols = [col for col in df.columns if 'CPU' in col and 'Temp' in col]
        temp_cols = sorted(temp_cols)
        
        # 定義顏色
        colors = plt.cm.tab10(np.linspace(0, 1, len(temp_cols)))
        
        for i, col in enumerate(temp_cols):
            temp_data = pd.to_numeric(df[col], errors='coerce')
            if not temp_data.isna().all():
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                ax.plot(df.index.total_seconds(), temp_data, 
                       label=core_name, color=colors[i], linewidth=2)
        
        ax.set_title("🔥 燒機測試 - CPU各核心溫度變化", fontsize=14, fontweight='bold')
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="CPU Cores", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
        
        if x_limits:
            ax.set_xlim(x_limits)
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def generate_burn_in_freq_chart(log_data: LogData, x_limits=None, y_limits=None):
        """生成燒機測試CPU頻率圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 獲取所有CPU頻率欄位
        freq_cols = [col for col in df.columns if 'CPU' in col and 'Freq' in col]
        freq_cols = sorted(freq_cols)
        
        # 定義顏色
        colors = plt.cm.Set1(np.linspace(0, 1, len(freq_cols)))
        
        for i, col in enumerate(freq_cols):
            freq_data = pd.to_numeric(df[col], errors='coerce')
            if not freq_data.isna().all():
                core_name = col.replace('BURN: ', '').replace('_', ' ')
                ax.plot(df.index.total_seconds(), freq_data, 
                       label=core_name, color=colors[i], linewidth=2)
        
        ax.set_title("⚡ 燒機測試 - CPU各核心頻率變化", fontsize=14, fontweight='bold')
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Frequency (MHz)", fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="CPU Cores", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
        
        if x_limits:
            ax.set_xlim(x_limits)
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        return fig

# =============================================================================
# 6. 燒機測試比較功能 (Burn-In Test Comparison)
# =============================================================================

class BurnInTestComparison:
    """燒機測試比較器"""
    
    @staticmethod
    def generate_comparison_table(log_data_list: List[LogData]) -> pd.DataFrame:
        """生成燒機測試比較表格"""
        burn_in_logs = [log for log in log_data_list if log.metadata.log_type == "Burn-In Test Log"]
        
        if len(burn_in_logs) < 2:
            return pd.DataFrame()
        
        comparison_data = []
        
        # 獲取所有CPU核心
        all_cores = set()
        for log_data in burn_in_logs:
            df = log_data.df
            temp_cols = [col for col in df.columns if 'CPU' in col and 'Temp' in col]
            for col in temp_cols:
                core_name = col.replace('BURN: ', '').replace('_Temp_C', '')
                all_cores.add(core_name)
        
        all_cores = sorted(all_cores)
        
        for core in all_cores:
            row_data = {'CPU Core': core}
            
            for i, log_data in enumerate(burn_in_logs):
                df = log_data.df
                test_name = f"Test {i+1} ({log_data.metadata.filename})"
                
                # 找對應的溫度欄位
                temp_col = f'BURN: {core}_Temp_C'
                if temp_col in df.columns:
                    temp_data = pd.to_numeric(df[temp_col], errors='coerce').dropna()
                    if len(temp_data) > 0:
                        max_temp = temp_data.max()
                        avg_temp = temp_data.mean()
                        row_data[f'{test_name} Max (°C)'] = f"{max_temp:.1f}"
                        row_data[f'{test_name} Avg (°C)'] = f"{avg_temp:.1f}"
                    else:
                        row_data[f'{test_name} Max (°C)'] = "N/A"
                        row_data[f'{test_name} Avg (°C)'] = "N/A"
                else:
                    row_data[f'{test_name} Max (°C)'] = "N/A"
                    row_data[f'{test_name} Avg (°C)'] = "N/A"
            
            comparison_data.append(row_data)
        
        return pd.DataFrame(comparison_data)

# =============================================================================
# 7. UI渲染層 (UI Rendering Layer)
# =============================================================================

class BurnInTestRenderer:
    """燒機測試UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render_controls(self, file_index=None):
        """渲染控制面板"""
        if file_index is None:
            file_index = getattr(st.session_state, 'current_file_index', 0)
        key_prefix = f"burn_{file_index}_"
        
        st.sidebar.markdown("### ⚙️ 燒機測試圖表設定")
        
        st.sidebar.markdown("#### 📈 圖表類型選擇")
        chart_type = st.sidebar.radio(
            "選擇圖表類型", 
            ["CPU溫度圖", "CPU頻率圖"],
            key=f"{key_prefix}chart_type"
        )
        
        st.sidebar.markdown("#### ⏱️ 時間範圍設定")
        time_min, time_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider(
            "選擇時間範圍 (秒)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
            step=1.0,
            key=f"{key_prefix}x_range"
        )
        
        st.sidebar.markdown("#### 📏 Y軸範圍設定")
        y_range_enabled = st.sidebar.checkbox("啟用Y軸範圍限制", key=f"{key_prefix}y_range_enabled")
        y_range = None
        if y_range_enabled:
            if chart_type == "CPU溫度圖":
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    y_min = st.number_input("Y軸最小值 (°C)", value=30.0, key=f"{key_prefix}y_min")
                with col2:
                    y_max = st.number_input("Y軸最大值 (°C)", value=100.0, key=f"{key_prefix}y_max")
            else:  # CPU頻率圖
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    y_min = st.number_input("Y軸最小值 (MHz)", value=0.0, key=f"{key_prefix}y_min")
                with col2:
                    y_max = st.number_input("Y軸最大值 (MHz)", value=3000.0, key=f"{key_prefix}y_max")
            
            y_range = (y_min, y_max)
        
        return chart_type, x_range, y_range
    
    def render_chart(self, chart_type, x_range, y_range):
        """渲染圖表"""
        st.markdown("### 📊 燒機測試監控圖表")
        
        if chart_type == "CPU溫度圖":
            chart = self.chart_gen.generate_burn_in_temp_chart(self.log_data, x_range, y_range)
        else:  # CPU頻率圖
            chart = self.chart_gen.generate_burn_in_freq_chart(self.log_data, x_range, y_range)
        
        if chart:
            st.pyplot(chart)
        else:
            st.warning("無法生成圖表，請檢查數據")
    
    def render_statistics(self, x_range):
        """渲染統計數據"""
        st.markdown("### 📈 燒機測試統計數據")
        
        temp_stats, freq_stats = self.stats_calc.calculate_burn_in_stats(self.log_data, x_range)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if temp_stats is not None and not temp_stats.empty:
                st.markdown("#### 🌡️ CPU溫度統計")
                st.dataframe(temp_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("無CPU溫度數據")
        
        with col2:
            if freq_stats is not None and not freq_stats.empty:
                st.markdown("#### ⚡ CPU頻率統計")
                st.dataframe(freq_stats, use_container_width=True, hide_index=True)
            else:
                st.warning("無CPU頻率數據")
    
    def render(self, file_index=None):
        """渲染完整UI"""
        st.markdown("""
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <h4>🔥 燒機測試Log 解析完成！</h4>
            <p>已識別為燒機測試數據，包含CPU溫度、頻率、GPU溫度等監控指標</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 顯示測試概況
        df = self.log_data.df
        cpu_count = len([col for col in df.columns if 'CPU' in col and 'Temp' in col])
        duration_minutes = self.log_data.df.index.total_seconds().max() / 60
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU核心數", f"{cpu_count}", help="監控的CPU核心數量")
        with col2:
            st.metric("測試時長", f"{duration_minutes:.1f} 分鐘", help="燒機測試總時長")
        with col3:
            st.metric("數據點數", f"{self.log_data.metadata.rows}", help="總記錄數據點")
        
        chart_type, x_range, y_range = self.render_controls(file_index)
        self.render_chart(chart_type, x_range, y_range)
        self.render_statistics(x_range)

# =============================================================================
# 8. UI工廠更新 (UI Factory Update)
# =============================================================================

class RendererFactory:
    """UI渲染器工廠"""
    
    @staticmethod
    def create_renderer(log_data: LogData):
        """根據log類型創建對應的渲染器"""
        log_type = log_data.metadata.log_type
        
        if log_type == "Burn-In Test Log":
            return BurnInTestRenderer(log_data)
        elif log_type == "GPUMon Log":
            # return GPUMonRenderer(log_data)  # 簡化示例
            return None
        elif log_type == "PTAT Log":
            # return PTATRenderer(log_data)  # 簡化示例
            return None
        elif log_type == "YOKOGAWA Log":
            # return YokogawaRenderer(log_data)  # 簡化示例
            return None
        else:
            return None

# =============================================================================
# 9. 主應用程式 (Main Application)
# =============================================================================

def display_version_info():
    """顯示版本資訊"""
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"""
        **當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**
        
        ### ✨ 主要功能
        
        - **🔥 Burn-In Test Log** - 燒機測試數據解析與視覺化 (新增)
        - **🎮 GPUMon Log** - GPU性能監控數據解析與視覺化
        - **🖥️ PTAT Log** - CPU性能監控數據解析與視覺化  
        - **📊 YOKOGAWA Log** - 多通道溫度記錄儀數據解析與視覺化
        - **📋 多檔案比較** - 燒機測試結果自動比較分析
        
        ### 🔥 燒機測試功能特色
        
        - **🌡️ CPU溫度監控** - 每個核心的溫度變化曲線
        - **⚡ CPU頻率監控** - 每個核心的頻率變化曲線
        - **📊 統計分析** - 最大、最小、平均溫度/頻率
        - **🔄 多檔案比較** - 自動整合多次燒機測試結果
        - **⏱️ 時間範圍選擇** - 可自定義分析時間區間
        """)

def main():
    """主程式 - v10.4.0 with Burn-In Test Support"""
    st.set_page_config(
        page_title="溫度數據視覺化平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS樣式
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .burn-in-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .comparison-table {
            font-size: 0.9em;
        }
        .comparison-table th {
            background-color: #f0f2f6;
            font-weight: bold;
            text-align: center;
        }
        .comparison-table td {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 標題
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 溫度數據視覺化平台</h1>
        <p>智能解析 YOKOGAWA、PTAT、GPUMon、燒機測試 Log 文件</p>
        <p><strong>{VERSION}</strong> | {VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    display_version_info()
    
    # 初始化解析器註冊系統
    parser_registry = ParserRegistry()
    parser_registry.register(BurnInTestParser())  # 新增燒機測試解析器
    parser_registry.register(GPUMonParser())
    parser_registry.register(PTATParser())
    parser_registry.register(YokogawaParser())  # 兜底解析器
    
    # 側邊欄
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx', 'log', 'txt'], 
        accept_multiple_files=True,
        help="v10.4.0 新增燒機測試支援，支援多檔案比較分析"
    )
    
    # 顯示訪問計數器
    display_visit_counter()
    
    if uploaded_files:
        # 顯示上傳檔案資訊
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 已上傳檔案")
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / 1024
            st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
        
        st.sidebar.markdown("---")
        
        # 解析檔案
        log_data_list = []
        for uploaded_file in uploaded_files:
            log_data = parser_registry.parse_file(uploaded_file)
            if log_data:
                log_data_list.append(log_data)
        
        if not log_data_list:
            st.error("❌ 無法解析任何檔案")
            return
        
        # 檢查是否有燒機測試檔案
        burn_in_logs = [log for log in log_data_list if log.metadata.log_type == "Burn-In Test Log"]
        
        if len(burn_in_logs) > 1:
            # 多個燒機測試檔案 - 顯示比較功能
            st.success(f"🔥 燒機測試比較模式：找到 {len(burn_in_logs)} 個燒機測試檔案")
            
            # 創建標籤頁
            tab_names = ["📋 比較分析"] + [f"🔥 測試 {i+1}" for i in range(len(burn_in_logs))]
            tabs = st.tabs(tab_names)
            
            # 比較分析標籤頁
            with tabs[0]:
                st.markdown("### 🔄 燒機測試比較分析")
                
                comparison_df = BurnInTestComparison.generate_comparison_table(burn_in_logs)
                if not comparison_df.empty:
                    st.markdown("#### 🌡️ CPU溫度比較表格")
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # 顯示比較摘要
                    st.markdown("#### 📊 比較摘要")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("測試數量", f"{len(burn_in_logs)}", help="比較的燒機測試數量")
                    with col2:
                        total_duration = sum([log.df.index.total_seconds().max() for log in burn_in_logs]) / 60
                        st.metric("總測試時長", f"{total_duration:.1f} 分鐘", help="所有測試的總時長")
                    with col3:
                        cpu_cores = len([col for col in burn_in_logs[0].df.columns if 'CPU' in col and 'Temp' in col])
                        st.metric("監控CPU核心", f"{cpu_cores}", help="監控的CPU核心數量")
                else:
                    st.warning("無法生成比較表格")
            
            # 各個測試的詳細分析
            for i, (tab, log_data) in enumerate(zip(tabs[1:], burn_in_logs)):
                with tab:
                    renderer = BurnInTestRenderer(log_data)
                    renderer.render(file_index=i)
        
        elif len(log_data_list) == 1:
            # 單檔案模式
            log_data = log_data_list[0]
            renderer = RendererFactory.create_renderer(log_data)
            
            if renderer:
                renderer.render(file_index=0)
            else:
                st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
        
        else:
            # 多檔案混合模式
            st.success(f"📊 多檔案分析模式：成功解析 {len(log_data_list)} 個檔案")
            
            # 創建標籤頁
            tab_names = []
            for i, log_data in enumerate(log_data_list):
                filename = log_data.metadata.filename
                log_type = log_data.metadata.log_type
                
                short_name = filename[:12] + "..." if len(filename) > 15 else filename
                
                if "Burn-In Test" in log_type:
                    tab_name = f"🔥 {short_name}"
                elif "GPUMon" in log_type:
                    tab_name = f"🎮 {short_name}"
                elif "PTAT" in log_type:
                    tab_name = f"🖥️ {short_name}"
                elif "YOKOGAWA" in log_type:
                    tab_name = f"📊 {short_name}"
                else:
                    tab_name = f"📄 {short_name}"
                
                tab_names.append(tab_name)
            
            tabs = st.tabs(tab_names)
            
            for i, (tab, log_data) in enumerate(zip(tabs, log_data_list)):
                with tab:
                    # 顯示檔案資訊
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #1f77b4;">
                        <h4>📁 檔案資訊</h4>
                        <p><strong>檔案名稱：</strong> {log_data.metadata.filename}</p>
                        <p><strong>檔案類型：</strong> {log_data.metadata.log_type}</p>
                        <p><strong>數據規模：</strong> {log_data.metadata.rows} 行 × {log_data.metadata.columns} 列</p>
                        <p><strong>檔案大小：</strong> {log_data.metadata.file_size_kb:.1f} KB</p>
                        <p><strong>時間範圍：</strong> {log_data.metadata.time_range}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    renderer = RendererFactory.create_renderer(log_data)
                    
                    if renderer:
                        renderer.render(file_index=i)
                    else:
                        st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
    
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件進行分析")
        
        st.markdown("""
        ### 📋 支援的檔案格式
        
        - **🔥 燒機測試 Log** - CPU/GPU燒機測試監控數據 (新增支援)
        - **🎮 GPUMon CSV** - GPU性能監控數據（溫度、功耗、頻率、使用率）
        - **🖥️ PTAT CSV** - CPU性能監控數據（頻率、功耗、溫度）
        - **📊 YOKOGAWA Excel/CSV** - 多通道溫度記錄儀數據
        
        ### 🔥 燒機測試特色功能
        
        - **🌡️ CPU溫度圖表** - 每個核心的溫度變化曲線
        - **⚡ CPU頻率圖表** - 每個核心的頻率變化曲線  
        - **📊 統計分析** - 最大、最小、平均溫度和頻率
        - **🔄 多檔案比較** - 自動整合多次燒機測試結果
        - **⏱️ 時間範圍控制** - 可自定義分析的時間區間
        """)

if __name__ == "__main__":
    main()
