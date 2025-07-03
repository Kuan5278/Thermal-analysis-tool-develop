# thermal_analysis_platform_v10.3.9_integrated.py
# 溫度數據視覺化平台 - v10.3.9 (新增System Log格式解析 + 指定佈局呈現 + 支援txt)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime, date, timedelta
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os

# 版本資訊
VERSION = "v10.3.9 Multi-File Analysis with SystemLog Support"
VERSION_DATE = "2025年7月"

# =============================================================================
# 0. 訪問計數器 (Visit Counter)
# =============================================================================

class VisitCounter:
    """訪問計數器"""
    
    def __init__(self, counter_file="visit_counter.json"):
        self.counter_file = counter_file
        self.data = self._load_counter()
    
    def _load_counter(self) -> dict:
        try:
            if os.path.exists(self.counter_file):
                with open(self.counter_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass # 發生任何錯誤都返回預設值
        return {"total_visits": 0, "daily_visits": {}, "first_visit": None, "last_visit": None}
    
    def _save_counter(self):
        try:
            with open(self.counter_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def increment_visit(self):
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        self.data["total_visits"] += 1
        self.data["daily_visits"][today] = self.data["daily_visits"].get(today, 0) + 1
        if self.data["first_visit"] is None: self.data["first_visit"] = now.isoformat()
        self.data["last_visit"] = now.isoformat()
        self._cleanup_old_records()
        self._save_counter()
    
    def _cleanup_old_records(self):
        try:
            today = date.today()
            cutoff_date = today - timedelta(days=30)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d")
            keys_to_remove = [k for k in self.data["daily_visits"].keys() if k < cutoff_str]
            for key in keys_to_remove: del self.data["daily_visits"][key]
        except Exception: pass
    
    def get_stats(self) -> dict:
        today_str = date.today().strftime("%Y-%m-%d")
        yesterday_str = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        recent_7_days = sum(self.data["daily_visits"].get((date.today() - timedelta(days=i)).strftime("%Y-%m-%d"), 0) for i in range(7))
        return {
            "total_visits": self.data["total_visits"],
            "today_visits": self.data["daily_visits"].get(today_str, 0),
            "yesterday_visits": self.data["daily_visits"].get(yesterday_str, 0),
            "recent_7_days": recent_7_days,
            "first_visit": self.data["first_visit"],
            "last_visit": self.data["last_visit"],
            "active_days": len(self.data["daily_visits"])
        }

def display_visit_counter():
    if 'visit_counter' not in st.session_state:
        st.session_state.visit_counter = VisitCounter()
        st.session_state.visit_counted = False
    if not st.session_state.visit_counted:
        st.session_state.visit_counter.increment_visit()
        st.session_state.visit_counted = True
    stats = st.session_state.visit_counter.get_stats()
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 使用統計")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💫 總訪問", value=f"{stats['total_visits']:,}")
            st.metric(label="📅 今日", value=f"{stats['today_visits']:,}", delta=f"+{stats['today_visits'] - stats['yesterday_visits']}" if stats['yesterday_visits'] > 0 else None)
        with col2:
            st.metric(label="📈 近7天", value=f"{stats['recent_7_days']:,}")
            st.metric(label="🗓️ 活躍天數", value=f"{stats['active_days']:,}")
        with st.expander("📋 詳細統計", expanded=False):
            if stats['first_visit']: st.write(f"🚀 **首次使用：** {datetime.fromisoformat(stats['first_visit']).strftime('%Y-%m-%d %H:%M')}")
            if stats['last_visit']: st.write(f"⏰ **最後使用：** {datetime.fromisoformat(stats['last_visit']).strftime('%Y-%m-%d %H:%M')}")
            st.write(f"📊 **平均每日：** {stats['total_visits'] / max(stats['active_days'], 1):.1f} 次")

# =============================================================================
# 1. 數據模型層 (Data Model Layer)
# =============================================================================
@dataclass
class LogMetadata:
    filename: str; log_type: str; rows: int; columns: int; time_range: str; file_size_kb: float

class LogData:
    def __init__(self, df: pd.DataFrame, metadata: LogMetadata):
        self.df = df; self.metadata = metadata; self._numeric_columns = None
    
    @property
    def numeric_columns(self) -> List[str]:
        if self._numeric_columns is None: self._numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        return self._numeric_columns
    
    def get_time_range(self) -> Tuple[float, float]:
        if self.df.empty: return (0.0, 0.0)
        return (0.0, self.df.index.total_seconds().max())
    
    def filter_by_time(self, x_limits: Tuple[float, float]):
        if x_limits is None: return self.df
        x_min_td = pd.to_timedelta(x_limits[0], unit='s'); x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        return self.df[(self.df.index >= x_min_td) & (self.df.index <= x_max_td)]

# =============================================================================
# 2. 解析器層 (Parser Layer)
# =============================================================================
class ParseLogger:
    def __init__(self): self.logs=[]; self.debug_logs=[]; self.success_logs=[]; self.error_logs=[]
    def info(self, m: str): self.logs.append(f"ℹ️ {m}")
    def debug(self, m: str): self.debug_logs.append(f"🔍 {m}")
    def success(self, m: str): self.success_logs.append(f"✅ {m}")
    def error(self, m: str): self.error_logs.append(f"❌ {m}")
    def warning(self, m: str): self.logs.append(f"⚠️ {m}")
    def show_summary(self, filename: str, log_type: str):
        if self.success_logs: st.success(f"✅ {log_type} 解析成功！")
        elif self.error_logs: st.error(f"❌ {filename} 解析失敗")
    def show_detailed_logs(self, filename: str):
        with st.expander(f"🔍 詳細解析日誌 - {filename}", expanded=False):
            if self.debug_logs: st.markdown("**🔍 調試信息：**"); [st.code(log, language=None) for log in self.debug_logs]
            if self.logs: st.markdown("**📋 解析過程：**"); [st.write(log) for log in self.logs]
            if self.success_logs: st.markdown("**✅ 成功信息：**"); [st.write(log) for log in self.success_logs]
            if self.error_logs: st.markdown("**❌ 錯誤信息：**"); [st.write(log) for log in self.error_logs]

class LogParser(ABC):
    def __init__(self): self.logger = ParseLogger()
    @abstractmethod
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool: pass
    @abstractmethod
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]: pass
    @property
    @abstractmethod
    def log_type(self) -> str: pass

class GPUMonParser(LogParser):
    @property
    def log_type(self) -> str: return "GPUMon Log"
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            first_content = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(100)])
            return any(['GPU Informations' in first_content, 'Iteration, Date, Timestamp' in first_content, 'Temperature GPU (C)' in first_content])
        except: return False
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        try:
            file_content.seek(0)
            content = file_content.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            header_row_index = next((i for i, line in enumerate(lines) if 'iteration' in line.lower() and 'date' in line.lower()), None)
            if header_row_index is None: self.logger.error("找不到GPUMon標題行"); return None
            
            headers = [h.strip() for h in lines[header_row_index].split(',')]
            data_rows = [line.split(',') for line in lines[header_row_index + 1:] if line.strip() and line.count(',') >= len(headers) - 5]
            if not data_rows: self.logger.error("無有效數據行"); return None

            df = pd.DataFrame(data_rows, columns=headers[:len(data_rows[0])])
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'].str.replace(r':(\d{3})$', r'.\1', regex=True), errors='coerce')
            df.dropna(subset=['DateTime'], inplace=True)
            df['time_index'] = df['DateTime'] - df['DateTime'].iloc[0]
            
            for col in df.columns:
                if col not in ['Date', 'Timestamp', 'DateTime', 'time_index', 'Iteration']:
                    df[col] = pd.to_numeric(df[col].replace(['N/A', 'n/a', ''], np.nan), errors='coerce')
            
            result_df = df.add_prefix('GPU: ').rename(columns={'GPU: time_index': 'time_index'}).set_index('time_index')
            metadata = LogMetadata(filename, self.log_type, *result_df.shape, f"{result_df.index.min()} 到 {result_df.index.max()}", len(content.encode('utf-8'))/1024)
            self.logger.success(f"GPUMon解析成功: {result_df.shape}")
            return LogData(result_df, metadata)
        except Exception as e: self.logger.error(f"GPUMon解析異常: {e}"); return None

class PTATParser(LogParser):
    @property
    def log_type(self) -> str: return "PTAT Log"
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            first_content = file_content.read(2000).decode('utf-8', errors='ignore')
            return 'MSR Package Temperature' in first_content or 'Version,Date,Time' in first_content
        except: return False
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        try:
            file_content.seek(0)
            df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
            df.columns = df.columns.str.strip()
            if 'Time' not in df.columns: self.logger.error("找不到時間欄位"); return None
            
            datetime_series = pd.to_datetime(df['Time'].astype(str).str.strip().str.replace(r':(\d{3})$', r'.\1', regex=True), format='%H:%M:%S.%f', errors='coerce')
            df = df[datetime_series.notna()].copy()
            if df.empty: self.logger.error("無有效時間數據"); return None

            valid_datetimes = datetime_series[datetime_series.notna()]
            df['time_index'] = valid_datetimes - valid_datetimes.iloc[0]
            result_df = df.add_prefix('PTAT: ').rename(columns={'PTAT: time_index': 'time_index'}).set_index('time_index')
            metadata = LogMetadata(filename, self.log_type, *result_df.shape, f"{result_df.index.min()} 到 {result_df.index.max()}", len(file_content.getvalue())/1024)
            self.logger.success(f"PTAT解析成功: {result_df.shape}")
            return LogData(result_df, metadata)
        except Exception as e: self.logger.error(f"PTAT解析失敗: {e}"); return None

class SystemLogParser(LogParser):
    @property
    def log_type(self) -> str: return "System Log"
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        try:
            file_content.seek(0)
            content_sample = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(100)])
            return bool(re.search(r'cpu\d+\s+freq:', content_sample) and re.search(r'cpu\d+\s+temp:', content_sample) and '***' in content_sample)
        except: return False
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        try:
            file_content.seek(0)
            content = file_content.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            records, current_record, temp_readings = [], {}, {}

            for line in lines:
                line = line.strip()
                if '**************************' in line:
                    if current_record and 'Timestamp' in current_record:
                        for key, values in temp_readings.items(): current_record[key] = sum(values) / len(values)
                        records.append(current_record)
                    current_record, temp_readings = {}, {}
                    ts_match = re.search(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})', line)
                    if ts_match: current_record['Timestamp'] = pd.to_datetime(ts_match.group(1))
                    continue
                if ':' in line:
                    key_raw, value_raw = [p.strip() for p in line.split(':', 1)]
                    try:
                        value = float(value_raw)
                        col_name_base = key_raw.replace(' ', '_').replace('cpul', 'cpu1')
                        if 'temp' in key_raw:
                            col_name = f"{col_name_base}_C"
                            if col_name not in temp_readings: temp_readings[col_name] = []
                            temp_readings[col_name].append(value / 1000.0)
                        elif 'freq' in key_raw: current_record[f"{col_name_base}_MHz"] = value / 1000.0
                        elif 'Board temperature' in key_raw: current_record['Board_Temp_C'] = value
                    except ValueError: pass
            
            if current_record and 'Timestamp' in current_record:
                for key, values in temp_readings.items(): current_record[key] = sum(values) / len(values)
                records.append(current_record)

            if not records: self.logger.error("未解析到任何有效數據"); return None
            df = pd.DataFrame(records).dropna(subset=['Timestamp']).sort_values('Timestamp')
            df['time_index'] = df['Timestamp'] - df['Timestamp'].iloc[0]
            result_df = df.add_prefix('SYSLOG: ').rename(columns={'SYSLOG: time_index': 'time_index'}).set_index('time_index')
            metadata = LogMetadata(filename, self.log_type, *result_df.shape, f"{result_df.index.min()} 到 {result_df.index.max()}", len(content.encode('utf-8'))/1024)
            self.logger.success(f"System Log解析成功: {result_df.shape}")
            return LogData(result_df, metadata)
        except Exception as e: self.logger.error(f"System Log解析異常: {e}"); return None

class YokogawaParser(LogParser):
    @property
    def log_type(self) -> str: return "YOKOGAWA Log"
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool: return True
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        try:
            is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
            read_func = pd.read_excel if is_excel else pd.read_csv
            
            header_row = next((h for h in range(50) if self._is_header(read_func, file_content, h)), 29 if is_excel else 0)
            file_content.seek(0)
            df = read_func(file_content, header=header_row, thousands=',')
            df.columns = df.columns.str.strip()
            time_column = next((c for c in df.columns if 'time' in str(c).lower() or 'date' in str(c).lower()), None)
            if not time_column: self.logger.error("YOKOGAWA找不到時間欄位"); return None
            
            try: df['time_index'] = pd.to_timedelta(df[time_column].astype(str).str.strip() + ':00')
            except: df['time_index'] = pd.to_datetime(df[time_column], errors='coerce') - pd.to_datetime(df[time_column], errors='coerce').iloc[0]
            df.dropna(subset=['time_index'], inplace=True)
            if not df.empty: df['time_index'] -= df['time_index'].iloc[0]

            result_df = df.add_prefix('YOKO: ').rename(columns={'YOKO: time_index': 'time_index'}).set_index('time_index')
            metadata = LogMetadata(filename, self.log_type, *result_df.shape, f"{result_df.index.min()} 到 {result_df.index.max()}", len(file_content.getvalue())/1024)
            self.logger.success(f"YOKOGAWA解析成功: {result_df.shape}")
            return LogData(result_df, metadata)
        except Exception as e: self.logger.error(f"YOKOGAWA解析失敗: {e}"); return None
    def _is_header(self, read_func, file_content, row_num):
        try:
            file_content.seek(0)
            df = read_func(file_content, header=row_num, nrows=1)
            return any('time' in str(c).lower() or 'date' in str(c).lower() for c in df.columns)
        except: return False

# =============================================================================
# 3. 解析器註冊系統 (Parser Registry)
# =============================================================================
class ParserRegistry:
    def __init__(self): self.parsers: List[LogParser] = []
    def register(self, parser: LogParser): self.parsers.append(parser)
    def parse_file(self, uploaded_file) -> Optional[LogData]:
        filename = uploaded_file.name; file_content = io.BytesIO(uploaded_file.getvalue())
        for parser in self.parsers:
            try:
                file_content.seek(0)
                if parser.can_parse(file_content, filename):
                    file_content.seek(0)
                    result = parser.parse(file_content, filename)
                    if result:
                        parser.logger.show_summary(filename, parser.log_type)
                        parser.logger.show_detailed_logs(filename)
                        return result
            except Exception: continue
        st.error(f"❌ 無法解析檔案 {filename}")
        return None

# =============================================================================
# 4. 統計計算層 (Statistics Layer)
# =============================================================================
class StatisticsCalculator:
    @staticmethod
    def calculate_gpumon_stats(log_data: LogData, x_limits=None):
        df = log_data.filter_by_time(x_limits); temp_stats, power_stats, freq_stats, util_stats = [], [], [], []
        for col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty: continue
            name = col.replace("GPU: ", "")
            stats = {'Max': f"{data.max():.2f}", 'Min': f"{data.min():.2f}", 'Avg': f"{data.mean():.2f}"}
            if 'Temperature GPU' in col: temp_stats.append({'Sensor': name, **stats})
            elif 'Power' in col or 'TGP' in col: power_stats.append({'Rail': name, **stats})
            elif 'Clock' in col: freq_stats.append({'Domain': name, **stats})
            elif 'Utilization' in col: util_stats.append({'Type': name, **stats})
        return pd.DataFrame(temp_stats), pd.DataFrame(power_stats), pd.DataFrame(freq_stats), pd.DataFrame(util_stats)

    @staticmethod
    def calculate_ptat_stats(log_data: LogData, x_limits=None):
        df = log_data.filter_by_time(x_limits); freq_stats, power_stats, temp_stats = [], [], []
        for col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if data.empty: continue
            name = col.replace('PTAT: ', '')
            stats = {'Max': f"{data.max():.2f}", 'Min': f"{data.min():.2f}", 'Avg': f"{data.mean():.2f}"}
            if 'frequency' in col.lower(): freq_stats.append({'Core': name, **stats})
            elif 'power' in col.lower(): power_stats.append({'Type': name, **stats})
            elif 'temperature' in col.lower(): temp_stats.append({'Type': name, **stats})
        return pd.DataFrame(freq_stats), pd.DataFrame(power_stats), pd.DataFrame(temp_stats)

    @staticmethod
    def calculate_temp_stats(log_data: LogData, x_limits=None):
        df = log_data.filter_by_time(x_limits); stats_data = []
        for col in df.select_dtypes(include=['number']).columns:
            if col.lower() in ['date', 'sec', 'rt', 'time']: continue
            data = pd.to_numeric(df[col], errors='coerce').dropna()
            if not data.empty:
                stats_data.append({'Channel': col.split(': ')[-1], 'Tmax (°C)': f"{data.max():.2f}", 'Tavg (°C)': f"{data.mean():.2f}"})
        return pd.DataFrame(stats_data)

# =============================================================================
# 5. Summary溫度整合表格生成器 (Temperature Summary Generator)
# =============================================================================
class TemperatureSummaryGenerator:
    @staticmethod
    def generate_summary_table(log_data_list: List[LogData]) -> pd.DataFrame:
        summary_data = []
        for log_data in log_data_list:
            df = log_data.df
            temp_cols = [c for c in df.select_dtypes(include=['number']).columns if any(k in c.lower() for k in ['temp', 'ch_'])]
            if "PTAT" in log_data.metadata.log_type: temp_cols = [c for c in temp_cols if 'msr' in c.lower()]
            
            for col in temp_cols:
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if not data.empty:
                    summary_data.append({
                        'Location': col.split(': ')[-1],
                        'Result (Case Temp)': f"{data.max():.1f}",
                        'Source File': log_data.metadata.filename,
                    })
        df_summary = pd.DataFrame(summary_data)
        if not df_summary.empty: df_summary.insert(0, 'Ch.', range(1, 1 + len(df_summary)))
        return df_summary

    @staticmethod
    def format_summary_table_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
        if summary_df.empty: return pd.DataFrame()
        display_df = summary_df[['Ch.', 'Location']].copy()
        display_df['Description'] = ''
        display_df['Spec location'] = ''
        display_df['spec'] = ''
        display_df['Ref Tc spec'] = ''
        display_df['Result (Case Temp)'] = summary_df['Result (Case Temp)']
        return display_df

# =============================================================================
# 6. 圖表生成層 (Chart Generation Layer)
# =============================================================================
class ChartGenerator:
    @staticmethod
    def generate_chart(log_data: LogData, left_col: str, right_col: str, x_limits, y_limits_left=None, y_limits_right=None):
        df = log_data.filter_by_time(x_limits)
        if df.empty or not left_col or left_col not in df.columns: return None
        
        fig, ax1 = plt.subplots(figsize=(10.2, 5.1))
        x_axis = df.index.total_seconds()
        
        ax1.plot(x_axis, pd.to_numeric(df[left_col], errors='coerce'), color='tab:blue', linewidth=1.5, label=left_col.split(': ')[-1])
        ax1.set_xlabel('Elapsed Time (seconds)'); ax1.set_ylabel(left_col.split(': ')[-1], color='tab:blue'); ax1.tick_params(axis='y', labelcolor='tab:blue')
        if y_limits_left: ax1.set_ylim(y_limits_left)
        
        if right_col and right_col != 'None' and right_col in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(x_axis, pd.to_numeric(df[right_col], errors='coerce'), color='tab:red', linewidth=1.5, label=right_col.split(': ')[-1])
            ax2.set_ylabel(right_col.split(': ')[-1], color='tab:red'); ax2.tick_params(axis='y', labelcolor='tab:red')
            if y_limits_right: ax2.set_ylim(y_limits_right)

        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        ax1.grid(True, linestyle='--', alpha=0.7)
        if x_limits: ax1.set_xlim(x_limits)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def generate_yokogawa_temp_chart(log_data: LogData, x_limits=None, y_limits=None):
        df = log_data.filter_by_time(x_limits)
        if df.empty: return None
        fig, ax = plt.subplots(figsize=(10.2, 5.1))
        cols_to_plot = [c for c in df.select_dtypes(include=['number']).columns if c.lower() not in ['date', 'sec', 'rt', 'time']][:15]
        for col in cols_to_plot:
            ax.plot(df.index.total_seconds(), pd.to_numeric(df[col], errors='coerce'), label=col.split(': ')[-1], linewidth=1)
        ax.set_title("YOKOGAWA All Channel Temperature Plot"); ax.set_xlabel("Elapsed Time (seconds)"); ax.set_ylabel("Temperature (°C)")
        ax.grid(True, linestyle='--', alpha=0.7); ax.legend(title="Channels", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
        if x_limits: ax.set_xlim(x_limits)
        if y_limits: ax.set_ylim(y_limits)
        fig.tight_layout(); return fig

# =============================================================================
# 7. UI渲染層 (UI Rendering Layer)
# =============================================================================
class BaseRenderer:
    def __init__(self, log_data: LogData):
        self.log_data = log_data; self.stats_calc = StatisticsCalculator(); self.chart_gen = ChartGenerator()
    def render_controls(self, key_prefix: str, y1_default_kw: str, y2_default_kw: str):
        st.sidebar.markdown("### ⚙️ 圖表設定")
        cols = self.log_data.numeric_columns
        y1_idx = next((i for i, c in enumerate(cols) if y1_default_kw in c.lower()), 0)
        y2_idx = next((i for i, c in enumerate(['None']+cols) if y2_default_kw in c.lower()), 0)
        y1 = st.sidebar.selectbox("📈 左側Y軸", cols, index=y1_idx, key=f"{key_prefix}_y1")
        y2 = st.sidebar.selectbox("📊 右側Y軸", ['None']+cols, index=y2_idx, key=f"{key_prefix}_y2")
        
        t_min, t_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider("⏱️ 時間範圍 (秒)", t_min, t_max, (t_min, t_max), 1.0, key=f"{key_prefix}_x")
        
        y1_range, y2_range = None, None
        if st.sidebar.checkbox("🔵 左Y軸範圍", key=f"{key_prefix}_y1en"): y1_range = (st.sidebar.number_input("Min", key=f"{key_prefix}_y1min"), st.sidebar.number_input("Max", key=f"{key_prefix}_y1max"))
        if y2 != 'None' and st.sidebar.checkbox("🔴 右Y軸範圍", key=f"{key_prefix}_y2en"): y2_range = (st.sidebar.number_input("Min", key=f"{key_prefix}_y2min"), st.sidebar.number_input("Max", key=f"{key_prefix}_y2max"))
        return y1, y2, x_range, y1_range, y2_range

class GPUMonRenderer(BaseRenderer):
    def render(self, file_index=None):
        st.markdown('<div class="gpumon-box"><h4>🎮 GPUMon Log 解析完成！</h4></div>', unsafe_allow_html=True)
        y1, y2, x, y1r, y2r = self.render_controls(f"gpu_{file_index}_", 'temperature', 'tgp')
        if y1:
            st.markdown("### 📊 性能監控圖表")
            chart = self.chart_gen.generate_chart(self.log_data, y1, y2, x, y1r, y2r)
            if chart: st.pyplot(chart)
            st.markdown("### 📈 統計數據")
            t, p, f, u = self.stats_calc.calculate_gpumon_stats(self.log_data, x)
            if not t.empty: st.markdown("#### 🌡️ 溫度"); st.dataframe(t, use_container_width=True, hide_index=True)
            if not p.empty: st.markdown("#### 🔋 功耗"); st.dataframe(p, use_container_width=True, hide_index=True)
            if not f.empty: st.markdown("#### ⚡ 頻率"); st.dataframe(f, use_container_width=True, hide_index=True)
            if not u.empty: st.markdown("#### 📊 使用率"); st.dataframe(u, use_container_width=True, hide_index=True)

class PTATRenderer(BaseRenderer):
    def render(self, file_index=None):
        st.markdown('<div class="info-box"><h4>🖥️ PTAT Log 解析完成！</h4></div>', unsafe_allow_html=True)
        y1, y2, x, y1r, y2r = self.render_controls(f"ptat_{file_index}_", 'temperature', 'package power')
        if y1:
            st.markdown("### 📊 CPU 性能圖表")
            chart = self.chart_gen.generate_chart(self.log_data, y1, y2, x, y1r, y2r)
            if chart: st.pyplot(chart)
            st.markdown("### 📈 統計數據")
            f, p, t = self.stats_calc.calculate_ptat_stats(self.log_data, x)
            if not f.empty: st.markdown("#### ⚡ 頻率"); st.dataframe(f, use_container_width=True, hide_index=True)
            if not p.empty: st.markdown("#### 🔋 功耗"); st.dataframe(p, use_container_width=True, hide_index=True)
            if not t.empty: st.markdown("#### 🌡️ 溫度"); st.dataframe(t, use_container_width=True, hide_index=True)

class YokogawaRenderer(BaseRenderer):
    def render(self, file_index=None):
        st.markdown('<div class="success-box"><h4>📊 YOKOGAWA Log 解析完成！</h4></div>', unsafe_allow_html=True)
        mode = st.sidebar.radio("📈 圖表模式", ["全通道溫度圖", "自定義雙軸圖"], key=f"yoko_{file_index}_mode")
        if mode == "全通道溫度圖":
            t_min, t_max = self.log_data.get_time_range()
            x_range = st.sidebar.slider("⏱️ 時間範圍 (秒)", t_min, t_max, (t_min, t_max), 1.0, key=f"yoko_{file_index}_x_all")
            y_range = (st.sidebar.number_input("Y Min", key=f"yoko_{file_index}_ymin"), st.sidebar.number_input("Y Max", value=100.0, key=f"yoko_{file_index}_ymax")) if st.sidebar.checkbox("Y軸範圍", key=f"yoko_{file_index}_yen_all") else None
            chart = self.chart_gen.generate_yokogawa_temp_chart(self.log_data, x_range, y_range)
            if chart: st.pyplot(chart)
        else:
            y1, y2, x, y1r, y2r = self.render_controls(f"yoko_{file_index}_", 'ch', 'ch')
            if y1:
                chart = self.chart_gen.generate_chart(self.log_data, y1, y2, x, y1r, y2r)
                if chart: st.pyplot(chart)
        st.markdown("### 📈 溫度統計數據")
        stats = self.stats_calc.calculate_temp_stats(self.log_data)
        if not stats.empty: st.dataframe(stats, use_container_width=True, hide_index=True)

class SystemLogRenderer:
    def __init__(self, log_data: LogData): self.log_data = log_data
    def render(self, file_index=None):
        st.markdown('<div class="info-box"><h4>📝 System Log 解析完成！</h4><p>已識別為多核心系統日誌，下方以摘要表格呈現統計結果。</p></div>', unsafe_allow_html=True)
        df = self.log_data.df
        if df.empty: st.warning("數據為空，無法生成摘要。"); return

        freq_stats, temp_stats = {}, {}
        for i in range(8):
            freq_col, temp_col = f'SYSLOG: cpu{i}_freq_MHz', f'SYSLOG: cpu{i}_temp_C'
            if freq_col in df.columns: freq_stats[f'CPU{i}'] = {'max': df[freq_col].max(), 'avg': df[freq_col].mean()}
            if temp_col in df.columns: temp_stats[f'CPU{i}'] = {'max': df[temp_col].max(), 'avg': df[temp_col].mean()}
        
        gpu_col = 'SYSLOG: gpu_temp_C'
        if gpu_col in df.columns: temp_stats['GPU'] = {'max': df[gpu_col].max(), 'avg': df[gpu_col].mean()}
        board_col = 'SYSLOG: Board_Temp_C'
        if board_col in df.columns: temp_stats['Board'] = {'max': df[board_col].max(), 'avg': df[board_col].mean()}

        # 建立Markdown表格
        table = "| **CPU 頻率 (MHz)** | | | | **溫度 (°C)** | | |\n"
        table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        table += "| **核心** | **最大值** | **平均值** | | **組件** | **最大值** | **平均值** |\n"

        cpu_cores = sorted([k for k in temp_stats.keys() if 'CPU' in k], key=lambda x: int(x.replace('CPU','')))
        other_components = sorted([k for k in temp_stats.keys() if 'CPU' not in k])
        
        max_rows = max(len(cpu_cores), len(cpu_cores) + len(other_components))

        for i in range(max_rows):
            # 頻率部分
            freq_part = "| |||"
            if i < len(cpu_cores):
                core = cpu_cores[i]
                if core in freq_stats:
                    f_max, f_avg = freq_stats[core]['max'], freq_stats[core]['avg']
                    freq_part = f"| {core} | {f_max:.1f} | {f_avg:.1f} |"
                else:
                    freq_part = f"| {core} | N/A | N/A |"

            # 溫度部分
            temp_part = "|||"
            if i < len(cpu_cores) + len(other_components):
                if i < len(cpu_cores):
                    comp = cpu_cores[i]
                else:
                    comp = other_components[i - len(cpu_cores)]
                
                if comp in temp_stats:
                    t_max, t_avg = temp_stats[comp]['max'], temp_stats[comp]['avg']
                    temp_part = f"| {comp} | {t_max:.1f} | {t_avg:.1f} |"

            table += freq_part + temp_part + '\n'

        st.markdown("### 📊 硬體監控日誌分析總結")
        st.markdown(table)
        st.markdown("--- \n **備註:** 平均值是根據日誌中所有相關數據點計算得出。")

class SummaryRenderer:
    def __init__(self, log_data_list: List[LogData]):
        self.log_data_list = log_data_list; self.summary_gen = TemperatureSummaryGenerator()
    def render(self):
        st.markdown("<h3>📋 溫度整合摘要報告</h3>", unsafe_allow_html=True)
        summary_df = self.summary_gen.generate_summary_table(self.log_data_list)
        if summary_df.empty: st.warning("⚠️ 沒有找到可用的溫度數據"); return
        
        with st.expander("📂 檔案來源詳情", expanded=False):
            for i, filename in enumerate(summary_df['Source File'].unique(), 1):
                st.write(f"**{i}.** `{filename}`")
        
        display_df = self.summary_gen.format_summary_table_for_display(summary_df)
        if not display_df.empty:
            html_table = display_df.to_html(index=False, border=1, classes='temp-table', justify='center')
            with st.expander("🔍 HTML表格預覽（可直接複製）", expanded=True):
                st.markdown(html_table, unsafe_allow_html=True)
                st.info("💡 提示：在上方表格上按住滑鼠左鍵拖拽選中整個表格，然後Ctrl+C複製，到Word中Ctrl+V貼上")

# =============================================================================
# 8. UI工廠 (UI Factory)
# =============================================================================
class RendererFactory:
    @staticmethod
    def create_renderer(log_data: LogData):
        log_type = log_data.metadata.log_type
        if log_type == "GPUMon Log": return GPUMonRenderer(log_data)
        elif log_type == "PTAT Log": return PTATRenderer(log_data)
        elif log_type == "System Log": return SystemLogRenderer(log_data)
        elif log_type == "YOKOGAWA Log": return YokogawaRenderer(log_data)
        else: return None

# =============================================================================
# 9. 主應用程式 (Main Application)
# =============================================================================
def display_version_info():
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"**版本：{VERSION}** | **日期：{VERSION_DATE}**")
        st.markdown("""
        - **新增** `System Log` 解析器與專用佈局
        - **新增** 支援 `.txt` 檔案上傳
        - **優化** 多檔案處理流程與UI
        """)

def main():
    st.set_page_config(page_title="溫度數據視覺化平台", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""<style> .main-header {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white;} .gpumon-box, .info-box, .success-box {padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid;} .gpumon-box {background-color: #fff3cd; border-color: #ffeaa7;} .info-box {background-color: #d1ecf1; border-color: #bee5eb;} .success-box {background-color: #d4edda; border-color: #c3e6cb;} </style>""", unsafe_allow_html=True)
    
    st.markdown(f'<div class="main-header"><h1>📊 溫度數據視覺化平台</h1><p>智能解析 YOKOGAWA、PTAT、GPUMon、System Log 文件</p><p><strong>{VERSION}</strong></p></div>', unsafe_allow_html=True)
    display_version_info()
    
    parser_registry = ParserRegistry()
    parser_registry.register(GPUMonParser())
    parser_registry.register(PTATParser())
    parser_registry.register(SystemLogParser())
    parser_registry.register(YokogawaParser())
    
    st.sidebar.markdown("### 🎛️ 控制面板")
    uploaded_files = st.sidebar.file_uploader("📁 上傳Log File (可多選)", type=['csv', 'xlsx', 'txt'], accept_multiple_files=True)
    
    display_visit_counter()
    
    if uploaded_files:
        st.sidebar.markdown("---"); st.sidebar.markdown("### 📂 已上傳檔案")
        for i, f in enumerate(uploaded_files, 1): st.sidebar.markdown(f"**{i}.** `{f.name}` ({len(f.getvalue())/1024:.1f} KB)")
        
        log_data_list = [p for f in uploaded_files if (p := parser_registry.parse_file(f))]
        if not log_data_list: st.error("❌ 無法解析任何檔案"); return
        
        if len(log_data_list) == 1:
            renderer = RendererFactory.create_renderer(log_data_list[0])
            if renderer: renderer.render(file_index=0)
        else:
            tab_names = ["📋 Summary"] + [f"{'🎮' if 'GPU' in d.metadata.log_type else '🖥️' if 'PTAT' in d.metadata.log_type else '📝' if 'System' in d.metadata.log_type else '📊'} {d.metadata.filename[:15]}..." for d in log_data_list]
            tabs = st.tabs(tab_names)
            with tabs[0]: SummaryRenderer(log_data_list).render()
            for i, (tab, log_data) in enumerate(zip(tabs[1:], log_data_list)):
                with tab:
                    st.markdown(f"<h4>📁 {log_data.metadata.filename} ({log_data.metadata.log_type})</h4>", unsafe_allow_html=True)
                    renderer = RendererFactory.create_renderer(log_data)
                    if renderer: renderer.render(file_index=i)
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件進行分析")
        st.markdown("""
        ### 📋 支援格式
        - **🎮 GPUMon CSV** | **🖥️ PTAT CSV** | **📝 System Log TXT** | **📊 YOKOGAWA Excel/CSV**
        ### ✨ 主要功能
        - **智能解析** | **多檔案分析** | **即時互動圖表** | **帶邊框表格複製**
        """)

if __name__ == "__main__":
    main()

