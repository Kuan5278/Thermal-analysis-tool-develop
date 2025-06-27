# thermal_analysis_platform_v10.3.8_optimized_fixed.py
# 溫度數據視覺化平台 - v10.3.8 多檔案獨立分析 + Summary整合版 (優化版 + 修復一鍵複製)

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
VERSION = "v10.3.8 Multi-File Analysis with Summary (Optimized + Fixed Copy with Borders)"
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
        
        # 顯示詳細信息
        with st.expander("📋 詳細統計", expanded=False):
            if stats['first_visit']:
                first_visit = datetime.fromisoformat(stats['first_visit'])
                st.write(f"🚀 **首次使用：** {first_visit.strftime('%Y-%m-%d %H:%M')}")
            
            if stats['last_visit']:
                last_visit = datetime.fromisoformat(stats['last_visit'])
                st.write(f"⏰ **最後使用：** {last_visit.strftime('%Y-%m-%d %H:%M')}")
            
            st.write(f"📊 **平均每日：** {stats['total_visits'] / max(stats['active_days'], 1):.1f} 次")
            
            # 顯示最近幾天的訪問趨勢
            recent_data = []
            for i in range(6, -1, -1):  # 最近7天，倒序
                check_date = date.today().replace(day=date.today().day-i) if date.today().day > i else date.today().replace(month=date.today().month-1, day=30-i+date.today().day)
                date_str = check_date.strftime("%Y-%m-%d")
                visits = st.session_state.visit_counter.data["daily_visits"].get(date_str, 0)
                recent_data.append({
                    'date': check_date.strftime("%m/%d"),
                    'visits': visits
                })
            
            if recent_data:
                st.write("📈 **最近7天趨勢：**")
                trend_text = " | ".join([f"{d['date']}: {d['visits']}" for d in recent_data])
                st.code(trend_text, language=None)

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
# 2. 解析器層 (Parser Layer) - 超簡潔版本
# =============================================================================

class ParseLogger:
    """解析日誌管理器 - 統一管理所有解析輸出"""
    
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

class GPUMonParser(LogParser):
    """GPUMon解析器 - 超簡潔版"""
    
    @property
    def log_type(self) -> str:
        return "GPUMon Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        """檢查是否為GPUMon格式"""
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
        """解析GPUMon檔案 - 靜默版本"""
        try:
            file_content.seek(0)
            content = file_content.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            
            self.logger.debug(f"檔案總行數: {len(lines)}")
            
            # 尋找標題行
            header_row_index = self._find_header_row(lines)
            if header_row_index is None:
                self.logger.error("找不到有效的標題行")
                return None
            
            # 解析數據
            df = self._parse_data_rows(lines, header_row_index)
            if df is None:
                self.logger.error("數據行解析失敗")
                return None
            
            # 處理時間
            df = self._process_time_data(df)
            if df is None:
                self.logger.error("時間數據處理失敗")
                return None
            
            # 數值轉換
            df = self._convert_numeric_columns(df)
            
            # 添加前綴並設置索引
            df = df.add_prefix('GPU: ')
            df.rename(columns={'GPU: time_index': 'time_index'}, inplace=True)
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
            
            self.logger.success(f"GPUMon解析成功！數據形狀: {result_df.shape}")
            return LogData(result_df, metadata)
            
        except Exception as e:
            self.logger.error(f"GPUMon解析異常: {e}")
            return None
    
    def _find_header_row(self, lines: List[str]) -> Optional[int]:
        """靜默尋找標題行"""
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if ('iteration' in line_lower and 'date' in line_lower and 'timestamp' in line_lower):
                self.logger.debug(f"找到標題行在第 {i+1} 行")
                return i
        
        # 備用搜尋
        for i, line in enumerate(lines):
            if line.count(',') > 10 and ('iteration' in line.lower() or 'gpu' in line.lower()):
                self.logger.debug(f"備用方式找到標題行在第 {i+1} 行")
                return i
        
        return None
    
    def _parse_data_rows(self, lines: List[str], header_row_index: int) -> Optional[pd.DataFrame]:
        """靜默解析數據行"""
        header_line = lines[header_row_index]
        self.logger.debug(f"解析標題行，長度: {len(header_line)}")
        
        headers = [h.strip() for h in header_line.split(',')]
        self.logger.debug(f"解析到 {len(headers)} 個欄位")
        
        data_rows = []
        valid_data_count = 0
        
        for i in range(header_row_index + 1, min(header_row_index + 100, len(lines))):
            line = lines[i].strip()
            if line and not line.startswith(','):
                try:
                    row_data = [cell.strip() for cell in line.split(',')]
                    if len(row_data) >= 3:
                        if (row_data[0].isdigit() or 
                            any(cell and cell != 'N/A' for cell in row_data[:5])):
                            data_rows.append(row_data)
                            valid_data_count += 1
                except Exception:
                    continue
        
        self.logger.debug(f"找到 {len(data_rows)} 行有效數據")
        
        if not data_rows:
            return None
        
        # 創建DataFrame
        max_cols = max(len(headers), max(len(row) for row in data_rows))
        
        while len(headers) < max_cols:
            headers.append(f'Column_{len(headers)}')
        
        for row in data_rows:
            while len(row) < max_cols:
                row.append('')
        
        df = pd.DataFrame(data_rows, columns=headers[:max_cols])
        self.logger.debug(f"DataFrame創建成功: {df.shape}")
        
        return df
    
    def _process_time_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """靜默處理時間數據"""
        try:
            if 'Date' in df.columns and 'Timestamp' in df.columns:
                self.logger.debug("處理時間格式: Date + Timestamp")
                
                df['Timestamp_fixed'] = df['Timestamp'].str.replace(r':(\d{3})$', r'.\1', regex=True)
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp_fixed'], errors='coerce')
                
            else:
                df['DateTime'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(range(len(df)), unit='s')
            
            valid_datetime_count = df['DateTime'].notna().sum()
            self.logger.debug(f"成功解析的時間點: {valid_datetime_count}/{len(df)}")
            
            if valid_datetime_count > 0:
                df['time_index'] = df['DateTime'] - df['DateTime'].iloc[0]
                valid_mask = df['time_index'].notna()
                df = df[valid_mask].copy()
                self.logger.debug(f"時間解析成功，最終數據: {len(df)} 行")
            else:
                df['time_index'] = pd.to_timedelta(range(len(df)), unit='s')
            
            return df
            
        except Exception as e:
            self.logger.warning(f"時間解析異常，使用默認時間: {e}")
            df['time_index'] = pd.to_timedelta(range(len(df)), unit='s')
            return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """靜默轉換數值型欄位"""
        numeric_count = 0
        for col in df.columns:
            if col not in ['Date', 'Timestamp', 'DateTime', 'time_index', 'Iteration']:
                try:
                    df[col] = df[col].replace(['N/A', 'n/a', '', ' '], np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_count += 1
                except:
                    pass
        
        self.logger.debug(f"轉換了 {numeric_count} 個數值欄位")
        return df

class PTATParser(LogParser):
    """PTAT解析器 - 超簡潔版"""
    
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
        try:
            file_content.seek(0)
            df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
            df.columns = df.columns.str.strip()
            
            if 'Time' not in df.columns:
                self.logger.error("找不到時間欄位")
                return None
            
            time_series = df['Time'].astype(str).str.strip()
            time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
            datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
            
            valid_times_mask = datetime_series.notna()
            df = df[valid_times_mask].copy()
            
            if df.empty:
                self.logger.error("沒有有效的時間數據")
                return None
            
            valid_datetimes = datetime_series[valid_times_mask]
            df['time_index'] = valid_datetimes - valid_datetimes.iloc[0]
            df = df.add_prefix('PTAT: ')
            df.rename(columns={'PTAT: time_index': 'time_index'}, inplace=True)
            
            result_df = df.set_index('time_index')
            
            # 創建元數據
            file_size_kb = len(file_content.getvalue()) / 1024
            time_range = f"{result_df.index.min()} 到 {result_df.index.max()}"
            
            metadata = LogMetadata(
                filename=filename,
                log_type=self.log_type,
                rows=result_df.shape[0],
                columns=result_df.shape[1],
                time_range=time_range,
                file_size_kb=file_size_kb
            )
            
            self.logger.success(f"PTAT解析成功！數據形狀: {result_df.shape}")
            return LogData(result_df, metadata)
            
        except Exception as e:
            self.logger.error(f"PTAT解析失敗: {e}")
            return None

class YokogawaParser(LogParser):
    """YOKOGAWA解析器 - v10.3.8 超簡潔版本"""
    
    @property
    def log_type(self) -> str:
        return "YOKOGAWA Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        # YOKOGAWA作為兜底解析器，總是返回True
        return True
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        self.logger.info(f"啟動YOKOGAWA解析器 (v10.3.8超簡潔版) - {filename}")
        
        try:
            is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
            read_func = pd.read_excel if is_excel else pd.read_csv
            
            self.logger.debug(f"檔案類型: {'Excel' if is_excel else 'CSV'}")
            
            # 動態搜索可能的 header 行
            possible_headers = self._find_possible_headers(file_content, is_excel, read_func)
            
            df = None
            found_time_col = None
            successful_header = None
            
            self.logger.debug(f"候選header行: {possible_headers}")
            
            for header_row in possible_headers:
                try:
                    file_content.seek(0)
                    df = read_func(file_content, header=header_row, thousands=',')
                    df.columns = df.columns.str.strip()
                    
                    self.logger.debug(f"嘗試header_row={header_row}, 形狀: {df.shape}")
                    
                    time_candidates = ['Time', 'TIME', 'time', 'Date', 'DATE', 'date', 
                                     'DateTime', 'DATETIME', 'datetime', '時間', '日期時間',
                                     'Timestamp', 'TIMESTAMP', 'timestamp']
                    
                    for candidate in time_candidates:
                        if candidate in df.columns:
                            found_time_col = candidate
                            successful_header = header_row
                            self.logger.debug(f"找到時間欄位: {candidate}")
                            break
                    
                    if found_time_col:
                        break
                        
                except Exception as e:
                    self.logger.debug(f"header_row={header_row} 失敗: {e}")
                    continue
            
            if df is None or found_time_col is None:
                self.logger.error("無法找到時間欄位")
                return None
            
            time_column = found_time_col
            self.logger.success(f"成功解析，header_row={successful_header}, 時間欄位='{time_column}'")
            self.logger.debug(f"DataFrame形狀: {df.shape}")
            
            # 動態重命名邏輯 - 靜默執行
            if is_excel:
                try:
                    ch_row_idx, tag_row_idx = self._find_ch_tag_rows(file_content, successful_header)
                    
                    if ch_row_idx is not None and tag_row_idx is not None:
                        self.logger.debug(f"找到CH行(第{ch_row_idx+1}行)和Tag行(第{tag_row_idx+1}行)")
                        
                        # 讀取CH行和Tag行
                        file_content.seek(0)
                        ch_row = pd.read_excel(file_content, header=None, skiprows=ch_row_idx, nrows=1).iloc[0]
                        file_content.seek(0)
                        tag_row = pd.read_excel(file_content, header=None, skiprows=tag_row_idx, nrows=1).iloc[0]
                        
                        # 執行重命名
                        df = self._perform_renaming(df, ch_row, tag_row)
                    else:
                        self.logger.info("未找到CH/Tag行，使用原始欄位名稱")
                        
                except Exception as e:
                    self.logger.warning(f"重命名過程異常: {e}")
            
            # 處理時間和完成解析
            result = self._process_time_and_finalize(df, time_column, file_content, filename)
            
            return result
            
        except Exception as e:
            self.logger.error(f"YOKOGAWA解析器異常: {e}")
            return None
    
    def _find_possible_headers(self, file_content: io.BytesIO, is_excel: bool, read_func) -> List[int]:
        """靜默搜索可能的header行"""
        if not is_excel:
            return [0, 1, 2]  # CSV 通常在前幾行
        
        possible_headers = []
        
        self.logger.debug("開始搜索header行...")
        
        # 第一階段：關鍵字搜索
        time_keywords = ['time', 'date', 'timestamp', '時間', '日期']
        
        for pos in range(0, 50):  # 搜索前50行
            try:
                file_content.seek(0)
                test_df = read_func(file_content, header=pos, nrows=1)
                columns_str = ' '.join(str(col).lower() for col in test_df.columns if pd.notna(col))
                
                # 檢查是否包含時間相關關鍵詞
                if any(keyword in columns_str for keyword in time_keywords):
                    possible_headers.append(pos)
                    found_keywords = [kw for kw in time_keywords if kw in columns_str]
                    self.logger.debug(f"第{pos+1}行包含時間關鍵詞: {found_keywords}")
                    
            except Exception:
                continue
        
        # 第二階段：結構搜索
        if not possible_headers:
            self.logger.debug("關鍵字搜索失敗，使用結構搜索")
            for pos in range(0, 50):
                try:
                    file_content.seek(0)
                    test_df = read_func(file_content, header=pos, nrows=1)
                    if test_df.shape[1] >= 5:  # 至少要有5個欄位
                        possible_headers.append(pos)
                        if len(possible_headers) >= 10:  # 最多找10個候選
                            break
                except Exception:
                    continue
        
        # 第三階段：預設值
        if not possible_headers:
            possible_headers = [29, 28, 30, 27, 26, 31, 32] if is_excel else [0, 1, 2]
            self.logger.debug("使用預設搜索範圍")
        
        self.logger.debug(f"找到 {len(possible_headers)} 個候選header行")
        return possible_headers
    
    def _find_ch_tag_rows(self, file_content: io.BytesIO, header_row: int) -> Tuple[Optional[int], Optional[int]]:
        """靜默尋找CH行和Tag行"""
        ch_row_idx = None
        tag_row_idx = None
        
        self.logger.debug(f"在header行({header_row+1})附近搜索CH和Tag行")
        
        # 搜索範圍
        search_range = range(max(0, header_row - 8), header_row + 1)
        
        # 分析所有候選行的內容
        row_analysis = []
        for idx in search_range:
            try:
                file_content.seek(0)
                test_row = pd.read_excel(file_content, header=None, skiprows=idx, nrows=1).iloc[0]
                
                # 分析這一行的內容
                ch_count = 0
                meaningful_tags = []
                
                for val in test_row:
                    if pd.isna(val) or str(val).strip() == '':
                        continue
                    else:
                        val_str = str(val).strip()
                        
                        if val_str.upper().startswith('CH'):
                            ch_count += 1
                        elif self._is_meaningful_tag(val):
                            meaningful_tags.append(val_str)
                
                analysis = {
                    'row_idx': idx,
                    'ch_count': ch_count,
                    'meaningful_tags': meaningful_tags,
                    'meaningful_count': len(meaningful_tags)
                }
                row_analysis.append(analysis)
                
                self.logger.debug(f"第{idx+1}行: CH={ch_count}, 用戶標籤={len(meaningful_tags)}")
                
            except Exception as e:
                self.logger.debug(f"第{idx+1}行分析失敗: {e}")
                continue
        
        # 尋找CH行
        for analysis in row_analysis:
            if analysis['ch_count'] >= 3:
                ch_row_idx = analysis['row_idx']
                self.logger.debug(f"找到CH行在第{ch_row_idx+1}行")
                break
        
        # 尋找Tag行
        if ch_row_idx is not None:
            tag_candidates = [a for a in row_analysis if a['row_idx'] != ch_row_idx and a['row_idx'] < header_row]
            
            best_tag_row = None
            max_tags = 0
            
            for candidate in tag_candidates:
                if candidate['meaningful_count'] > max_tags:
                    max_tags = candidate['meaningful_count']
                    best_tag_row = candidate
            
            if best_tag_row:
                tag_row_idx = best_tag_row['row_idx']
                self.logger.debug(f"找到Tag行在第{tag_row_idx+1}行(含{max_tags}個用戶標籤)")
        
        return ch_row_idx, tag_row_idx
    
    def _is_meaningful_tag(self, tag_val) -> bool:
        """判斷Tag值是否有意義（用戶自定義代號）"""
        if pd.isna(tag_val):
            return False
            
        tag_str = str(tag_val).strip()
        
        # 排除空值
        if tag_str in ['', 'nan', 'NaN', 'None']:
            return False
            
        # 排除單獨的 "Tag"
        if tag_str.upper() == 'TAG':
            return False
            
        # 排除系統標題詞
        system_titles = ['CHANNEL', 'CH', 'POINT', 'TEMP', 'SENSOR']
        if tag_str.upper() in system_titles:
            return False
            
        # 字母+數字組合，很可能是用戶標籤（如 U5, U19, L8）
        if len(tag_str) <= 4 and any(c.isalpha() for c in tag_str) and any(c.isdigit() for c in tag_str):
            return True
            
        # 包含下劃線，很可能是用戶標籤（如 CPU_Tc）
        if '_' in tag_str:
            return True
            
        # 排除看起來像測量數據的數字
        try:
            float_val = float(tag_str)
            if (0 <= float_val <= 200 and '.' in tag_str and len(tag_str) > 4):
                return False
            elif len(tag_str) <= 3:
                return True
        except ValueError:
            pass
            
        # 其他情況，長度大於1就認為是有意義的
        if len(tag_str) >= 2:
            return True
            
        return False
    
    def _perform_renaming(self, df: pd.DataFrame, ch_row: pd.Series, tag_row: pd.Series) -> pd.DataFrame:
        """靜默執行重命名邏輯"""
        self.logger.debug("開始智能重命名處理")
        
        # 保護關鍵欄位
        protected_columns = {
            'Date', 'TIME', 'Time', 'time', 'DATE', 'date',
            'DateTime', 'DATETIME', 'datetime', 
            'Timestamp', 'TIMESTAMP', 'timestamp',
            'sec', 'SEC', 'RT', 'rt', '時間', '日期時間'
        }
        
        new_column_names = {}
        tag_used = 0
        ch_used = 0
        protected_count = 0
        original_kept = 0
        
        for i, original_col in enumerate(df.columns):
            # 保護關鍵欄位
            if original_col in protected_columns:
                final_name = original_col
                protected_count += 1
                new_column_names[original_col] = final_name
                continue
            
            # 獲取Tag值
            tag_name = ""
            if i < len(tag_row):
                tag_val = tag_row.iloc[i]
                if self._is_meaningful_tag(tag_val):
                    tag_name = str(tag_val).strip()
            
            # 獲取CH值
            ch_name = ""
            if i < len(ch_row):
                ch_val = ch_row.iloc[i]
                if pd.notna(ch_val) and str(ch_val).strip().upper().startswith('CH'):
                    ch_name = str(ch_val).strip()
            
            # 決定最終名稱
            if tag_name:
                final_name = tag_name
                tag_used += 1
            elif ch_name:
                final_name = ch_name
                ch_used += 1
            else:
                final_name = original_col
                original_kept += 1
            
            new_column_names[original_col] = final_name
        
        # 執行重命名
        df.rename(columns=new_column_names, inplace=True)
        
        self.logger.debug(f"重命名完成: Tag={tag_used}, CH={ch_used}, 保護={protected_count}, 原名={original_kept}")
        
        return df
    
    def _process_time_and_finalize(self, df: pd.DataFrame, time_column: str, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        """處理時間並完成解析"""
        self.logger.debug("處理時間數據")
        time_series = df[time_column].astype(str).str.strip()
        
        try:
            df['time_index'] = pd.to_timedelta(time_series + ':00').fillna(pd.to_timedelta('00:00:00'))
            if df['time_index'].isna().all():
                raise ValueError("Timedelta 轉換失敗")
            self.logger.debug("時間解析成功 (Timedelta格式)")
        except:
            try:
                datetime_series = pd.to_datetime(time_series, format='%H:%M:%S', errors='coerce')
                if datetime_series.notna().sum() == 0:
                    datetime_series = pd.to_datetime(time_series, errors='coerce')
                df['time_index'] = datetime_series - datetime_series.iloc[0]
                self.logger.debug("時間解析成功 (DateTime格式)")
            except Exception as e:
                self.logger.error(f"時間解析失敗: {e}")
                return None
        
        valid_times_mask = df['time_index'].notna()
        if valid_times_mask.sum() == 0:
            self.logger.error("沒有有效的時間數據")
            return None
        
        df = df[valid_times_mask].copy()
        
        if len(df) > 0:
            start_time = df['time_index'].iloc[0]
            df['time_index'] = df['time_index'] - start_time
        
        # 數值轉換
        numeric_columns = df.select_dtypes(include=['number']).columns
        numeric_converted = len(numeric_columns)
        
        self.logger.debug(f"數值轉換完成，處理了 {numeric_converted} 個欄位")
        
        # 添加前綴
        df = df.add_prefix('YOKO: ')
        df.rename(columns={'YOKO: time_index': 'time_index'}, inplace=True)
        
        result_df = df.set_index('time_index')
        
        # 創建元數據
        file_size_kb = len(file_content.getvalue()) / 1024
        time_range = f"{result_df.index.min()} 到 {result_df.index.max()}"
        
        metadata = LogMetadata(
            filename=filename,
            log_type=self.log_type,
            rows=result_df.shape[0],
            columns=result_df.shape[1],
            time_range=time_range,
            file_size_kb=file_size_kb
        )
        
        self.logger.success(f"YOKOGAWA v10.3.8 解析完成！數據形狀: {result_df.shape}")
        
        return LogData(result_df, metadata)

# =============================================================================
# 3. 解析器註冊系統 (Parser Registry) - 超簡潔版
# =============================================================================

class ParserRegistry:
    """解析器註冊系統 - 超簡潔版"""
    
    def __init__(self):
        self.parsers: List[LogParser] = []
    
    def register(self, parser: LogParser):
        """註冊解析器"""
        self.parsers.append(parser)
    
    def parse_file(self, uploaded_file) -> Optional[LogData]:
        """解析檔案，自動選擇合適的解析器 - 靜默版本"""
        filename = uploaded_file.name
        file_content = io.BytesIO(uploaded_file.getvalue())
        is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
        
        # 創建一個臨時的日誌收集器來顯示解析摘要
        parsing_summary = {"attempted": [], "successful": None, "failed": []}
        
        for parser in self.parsers:
            try:
                file_content.seek(0)
                parsing_summary["attempted"].append(parser.log_type)
                
                if parser.can_parse(file_content, filename):
                    file_content.seek(0)
                    result = parser.parse(file_content, filename)
                    if result is not None:
                        parsing_summary["successful"] = parser.log_type
                        # 顯示解析摘要
                        parser.logger.show_summary(filename, parser.log_type)
                        # 顯示詳細日誌（摺疊）
                        parser.logger.show_detailed_logs(filename)
                        return result
                    else:
                        parsing_summary["failed"].append(parser.log_type)
            except Exception as e:
                parsing_summary["failed"].append(f"{parser.log_type} (異常: {str(e)[:50]})")
                continue
        
        # 如果所有解析器都失敗
        st.error(f"❌ 無法解析檔案 {filename}")
        with st.expander(f"🔍 解析失敗詳情 - {filename}", expanded=False):
            st.write(f"**嘗試的解析器:** {', '.join(parsing_summary['attempted'])}")
            if parsing_summary["failed"]:
                st.write(f"**失敗的解析器:** {', '.join(parsing_summary['failed'])}")
            st.write("**建議:** 確認檔案格式是否正確，或聯繫技術支援")
        
        return None

# =============================================================================
# 4. 統計計算層 (Statistics Layer)
# =============================================================================

class StatisticsCalculator:
    """統計計算器"""
    
    @staticmethod
    def calculate_gpumon_stats(log_data: LogData, x_limits=None):
        """計算GPUMon統計數據"""
        df = log_data.filter_by_time(x_limits)
        if df.empty:
            return None, None, None, None
        
        # GPU溫度統計
        temp_stats = []
        temp_cols = [col for col in df.columns if 'Temperature' in col and 'GPU' in col]
        
        for col in temp_cols:
            temp_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(temp_data) > 0:
                temp_stats.append({
                    'Temperature Sensor': col.replace('GPU: ', ''),
                    'Max (°C)': f"{temp_data.max():.2f}",
                    'Min (°C)': f"{temp_data.min():.2f}",
                    'Avg (°C)': f"{temp_data.mean():.2f}"
                })
        
        temp_df = pd.DataFrame(temp_stats) if temp_stats else None
        
        # GPU功耗統計
        power_stats = []
        target_power_items = ['NVVDD', 'FBVDD', 'TGP']
        
        for target_item in target_power_items:
            matching_cols = [col for col in df.columns if target_item in col and ('Power' in col or 'TGP' in col)]
            
            for col in matching_cols:
                power_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(power_data) > 0:
                    display_name = col.replace('GPU: ', '')
                    if 'NVVDD' in col:
                        display_name = 'NVVDD Power'
                    elif 'FBVDD' in col:
                        display_name = 'FBVDD Power'
                    elif 'TGP' in col:
                        display_name = 'TGP (W)'
                    
                    power_stats.append({
                        'Power Rail': display_name,
                        'Max (W)': f"{power_data.max():.2f}",
                        'Min (W)': f"{power_data.min():.2f}",
                        'Avg (W)': f"{power_data.mean():.2f}"
                    })
                    break
        
        power_df = pd.DataFrame(power_stats) if power_stats else None
        
        # GPU頻率統計
        freq_stats = []
        freq_cols = [col for col in df.columns if 'Clock' in col and any(x in col for x in ['GPC', 'Memory'])]
        
        for col in freq_cols:
            freq_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(freq_data) > 0:
                freq_stats.append({
                    'Clock Domain': col.replace('GPU: ', ''),
                    'Max (MHz)': f"{freq_data.max():.0f}",
                    'Min (MHz)': f"{freq_data.min():.0f}",
                    'Avg (MHz)': f"{freq_data.mean():.0f}"
                })
        
        freq_df = pd.DataFrame(freq_stats) if freq_stats else None
        
        # GPU使用率統計
        util_stats = []
        util_cols = [col for col in df.columns if 'Utilization' in col]
        
        for col in util_cols:
            util_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(util_data) > 0:
                util_stats.append({
                    'Utilization Type': col.replace('GPU: ', ''),
                    'Max (%)': f"{util_data.max():.1f}",
                    'Min (%)': f"{util_data.min():.1f}",
                    'Avg (%)': f"{util_data.mean():.1f}"
                })
        
        util_df = pd.DataFrame(util_stats) if util_stats else None
        
        return temp_df, power_df, freq_df, util_df
    
    @staticmethod
    def calculate_ptat_stats(log_data: LogData, x_limits=None):
        """計算PTAT統計數據"""
        df = log_data.filter_by_time(x_limits)
        if df.empty:
            return None, None, None
        
        # CPU Core Frequency 統計
        freq_stats = []
        freq_cols = [col for col in df.columns if 'frequency' in col.lower() and 'core' in col.lower()]
        
        lfm_value = "N/A"
        hfm_value = "N/A"
        
        for col in df.columns:
            if 'lfm' in col.lower():
                lfm_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(lfm_data) > 0:
                    lfm_value = f"{lfm_data.iloc[0]:.0f} MHz"
            elif 'hfm' in col.lower():
                hfm_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(hfm_data) > 0:
                    hfm_value = f"{hfm_data.iloc[0]:.0f} MHz"
        
        if lfm_value == "N/A" or hfm_value == "N/A":
            all_freq_data = []
            for col in freq_cols:
                freq_data = pd.to_numeric(df[col], errors='coerce').dropna()
                all_freq_data.extend(freq_data.tolist())
            
            if all_freq_data:
                if lfm_value == "N/A":
                    lfm_value = f"{min(all_freq_data):.0f} MHz (估算)"
                if hfm_value == "N/A":
                    hfm_value = f"{max(all_freq_data):.0f} MHz (估算)"
        
        for col in freq_cols:
            freq_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(freq_data) > 0:
                freq_stats.append({
                    'Core': col.replace('PTAT: ', ''),
                    'Max (MHz)': f"{freq_data.max():.0f}",
                    'Min (MHz)': f"{freq_data.min():.0f}",
                    'Avg (MHz)': f"{freq_data.mean():.0f}"
                })
        
        if freq_stats:
            freq_stats.append({
                'Core': '--- 參考值 ---',
                'Max (MHz)': '',
                'Min (MHz)': '',
                'Avg (MHz)': ''
            })
            freq_stats.append({
                'Core': 'LFM (Low Freq Mode)',
                'Max (MHz)': lfm_value,
                'Min (MHz)': '',
                'Avg (MHz)': ''
            })
            freq_stats.append({
                'Core': 'HFM (High Freq Mode)',
                'Max (MHz)': hfm_value,
                'Min (MHz)': '',
                'Avg (MHz)': ''
            })
        
        freq_df = pd.DataFrame(freq_stats) if freq_stats else None
        
        # Package Power 統計
        power_stats = []
        target_power_items = [
            ('IA', 'IA Power'),
            ('GT', 'GT Power'), 
            ('Rest of package', 'Rest of Package Power'),
            ('Package', 'Package Power')
        ]
        
        for search_term, display_name in target_power_items:
            matching_cols = []
            for col in df.columns:
                col_lower = col.lower()
                search_lower = search_term.lower()
                
                if search_term == 'IA':
                    if 'ia' in col_lower and 'power' in col_lower and 'via' not in col_lower:
                        matching_cols.append(col)
                elif search_term == 'GT':
                    if 'gt' in col_lower and 'power' in col_lower and 'tgp' not in col_lower:
                        matching_cols.append(col)
                elif search_term == 'Rest of package':
                    if 'rest' in col_lower and 'package' in col_lower and 'power' in col_lower:
                        matching_cols.append(col)
                elif search_term == 'Package':
                    if 'package' in col_lower and 'power' in col_lower and 'rest' not in col_lower:
                        matching_cols.append(col)
            
            if matching_cols:
                col = matching_cols[0]
                power_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(power_data) > 0:
                    power_stats.append({
                        'Power Type': display_name,
                        'Max (W)': f"{power_data.max():.2f}",
                        'Min (W)': f"{power_data.min():.2f}",
                        'Avg (W)': f"{power_data.mean():.2f}"
                    })
        
        power_df = pd.DataFrame(power_stats) if power_stats else None
        
        # MSR Package Temperature 統計
        temp_stats = []
        temp_cols = [col for col in df.columns if 'temperature' in col.lower() and 'package' in col.lower() and 'msr' in col.lower()]
        
        for col in temp_cols:
            temp_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(temp_data) > 0:
                temp_stats.append({
                    'Temperature Type': col.replace('PTAT: ', ''),
                    'Max (°C)': f"{temp_data.max():.2f}",
                    'Min (°C)': f"{temp_data.min():.2f}",
                    'Avg (°C)': f"{temp_data.mean():.2f}"
                })
        
        temp_df = pd.DataFrame(temp_stats) if temp_stats else None
        
        return freq_df, power_df, temp_df
    
    @staticmethod
    def calculate_temp_stats(log_data: LogData, x_limits=None):
        """計算溫度統計數據"""
        df = log_data.filter_by_time(x_limits)
        if df.empty:
            return pd.DataFrame()
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        temp_cols = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
        
        stats_data = []
        for col in temp_cols:
            y_data = pd.to_numeric(df[col], errors='coerce')
            if not y_data.isna().all():
                t_max = y_data.max()
                t_avg = y_data.mean()
                
                display_name = col
                if display_name.startswith('YOKO: '):
                    display_name = display_name.replace('YOKO: ', '')
                elif display_name.startswith('PTAT: '):
                    display_name = display_name.replace('PTAT: ', '')
                elif display_name.startswith('GPU: '):
                    display_name = display_name.replace('GPU: ', '')
                
                if display_name.lower() in ['sec', 'time', 'rt', 'date']:
                    continue
                
                stats_data.append({
                    '通道名稱': display_name,
                    'Tmax (°C)': f"{t_max:.2f}" if pd.notna(t_max) else "N/A",
                    'Tavg (°C)': f"{t_avg:.2f}" if pd.notna(t_avg) else "N/A"
                })
        
        return pd.DataFrame(stats_data)

# =============================================================================
# 5. Summary溫度整合表格生成器 (Temperature Summary Generator) - 優化版
# =============================================================================

class TemperatureSummaryGenerator:
    """溫度整合摘要生成器 - v10.3.8優化版"""
    
    @staticmethod
    def generate_summary_table(log_data_list: List[LogData]) -> pd.DataFrame:
        """生成溫度摘要表格，按照用戶提供的格式"""
        summary_data = []
        ch_number = 1
        
        for log_data in log_data_list:
            df = log_data.df
            log_type = log_data.metadata.log_type
            filename = log_data.metadata.filename
            
            # 獲取所有數值型欄位
            numeric_cols = df.select_dtypes(include=['number']).columns
            temp_cols = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
            
            # 針對PTAT log特殊處理 - 只保留MSR Package Temperature
            if "PTAT" in log_type:
                temp_cols = [col for col in temp_cols if 'msr' in col.lower() and 'package' in col.lower() and 'temperature' in col.lower()]
            
            for col in temp_cols:
                temp_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(temp_data) > 0:
                    max_temp = temp_data.max()
                    
                    # 清理欄位名稱
                    clean_col_name = col
                    if clean_col_name.startswith('YOKO: '):
                        clean_col_name = clean_col_name.replace('YOKO: ', '')
                    elif clean_col_name.startswith('PTAT: '):
                        clean_col_name = clean_col_name.replace('PTAT: ', '')
                    elif clean_col_name.startswith('GPU: '):
                        clean_col_name = clean_col_name.replace('GPU: ', '')
                    
                    # 跳過非溫度相關欄位
                    if clean_col_name.lower() in ['sec', 'time', 'rt', 'date', 'iteration']:
                        continue
                    
                    # 根據不同log類型設定描述
                    description = ""
                    if "GPU" in log_type:
                        if "Temperature" in clean_col_name:
                            description = "GPU Temperature"
                    elif "PTAT" in log_type:
                        if "MSR" in clean_col_name and "Package" in clean_col_name:
                            description = "CPU MSR Package Temperature"
                    else:  # YOKOGAWA或其他
                        # 根據欄位名稱推測類型
                        if any(keyword in clean_col_name.upper() for keyword in ['CPU', 'PROCESSOR']):
                            description = "CPU"
                        elif any(keyword in clean_col_name.upper() for keyword in ['SSD', 'STORAGE']):
                            description = "SSD"
                        elif any(keyword in clean_col_name.upper() for keyword in ['DDR', 'MEMORY', 'RAM']):
                            description = "Memory"
                        elif any(keyword in clean_col_name.upper() for keyword in ['WIFI', 'WIRELESS']):
                            description = "WIFI"
                        else:
                            description = ""
                    
                    # 格式化溫度值
                    if max_temp > 200:  # 可能是毫度或其他單位
                        formatted_temp = f"{max_temp/1000:.1f}" if max_temp > 1000 else f"{max_temp:.1f}"
                    else:
                        formatted_temp = f"{max_temp:.1f}"
                    
                    # 所有spec相關欄位都留空
                    summary_data.append({
                        'Ch.': ch_number,
                        'Location': clean_col_name,
                        'Description': description,
                        'Spec location': "",  # 留空給用戶填寫
                        'spec': "",  # 留空給用戶填寫
                        'Ref Tc spec': "",  # 留空給用戶填寫
                        'Result (Case Temp)': formatted_temp,
                        'Source File': filename,
                        'Log Type': log_type
                    })
                    
                    ch_number += 1
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def format_summary_table_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
        """格式化表格以符合顯示要求"""
        if summary_df.empty:
            return pd.DataFrame()
        
        # 創建顯示用的DataFrame，不包含Source File和Log Type
        display_df = summary_df[['Ch.', 'Location', 'Description', 'Spec location', 'spec', 'Ref Tc spec', 'Result (Case Temp)']].copy()
        
        return display_df
    
    @staticmethod
    def get_summary_statistics(summary_df: pd.DataFrame) -> dict:
        """獲取摘要統計信息"""
        if summary_df.empty:
            return {}
        
        try:
            # 轉換溫度為數值
            temps = pd.to_numeric(summary_df['Result (Case Temp)'], errors='coerce').dropna()
            
            stats = {
                'total_channels': len(summary_df),
                'max_temp': temps.max() if len(temps) > 0 else 0,
                'min_temp': temps.min() if len(temps) > 0 else 0,
                'avg_temp': temps.mean() if len(temps) > 0 else 0,
                'files_analyzed': summary_df['Source File'].nunique() if 'Source File' in summary_df.columns else 0,
                'log_types': summary_df['Log Type'].unique().tolist() if 'Log Type' in summary_df.columns else []
            }
            
            return stats
        except Exception:
            return {
                'total_channels': len(summary_df),
                'max_temp': 0,
                'min_temp': 0,
                'avg_temp': 0,
                'files_analyzed': 0,
                'log_types': []
            }

# =============================================================================
# 6. 圖表生成層 (Chart Generation Layer)
# =============================================================================

class ChartGenerator:
    """圖表生成器"""
    
    @staticmethod
    def generate_gpumon_chart(log_data: LogData, left_col: str, right_col: str, x_limits, left_y_limits=None, right_y_limits=None):
        """生成GPUMon專用圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty or not left_col or left_col not in df.columns:
            return None
        if right_col and right_col != 'None' and right_col not in df.columns:
            return None
        
        df_chart = df.copy()
        df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
        if right_col and right_col != 'None':
            df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')
        
        fig, ax1 = plt.subplots(figsize=(10.2, 5.1))
        
        title = f'GPUMon: {left_col.replace("GPU: ", "")} {"& " + right_col.replace("GPU: ", "") if right_col and right_col != "None" else ""}'
        plt.title(title, fontsize=14, fontweight='bold')
        
        x_axis_seconds = df_chart.index.total_seconds()
        color = 'tab:orange'
        ax1.set_xlabel('Elapsed Time (seconds)', fontsize=11)
        ax1.set_ylabel(left_col.replace("GPU: ", ""), color=color, fontsize=11)
        ax1.plot(x_axis_seconds, df_chart['left_val'], color=color, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        if left_y_limits:
            ax1.set_ylim(left_y_limits)
        
        if right_col and right_col != 'None':
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel(right_col.replace("GPU: ", ""), color=color, fontsize=11)
            ax2.plot(x_axis_seconds, df_chart['right_val'], color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            
            if right_y_limits:
                ax2.set_ylim(right_y_limits)
        
        if x_limits:
            ax1.set_xlim(x_limits)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def generate_flexible_chart(log_data: LogData, left_col: str, right_col: str, x_limits, left_y_limits=None, right_y_limits=None):
        """生成靈活的雙軸圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty or not left_col or left_col not in df.columns:
            return None
        if right_col and right_col != 'None' and right_col not in df.columns:
            return None
        
        df_chart = df.copy()
        df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
        if right_col and right_col != 'None':
            df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')
        
        fig, ax1 = plt.subplots(figsize=(10.2, 5.1))
        plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=14, fontweight='bold')
        
        x_axis_seconds = df_chart.index.total_seconds()
        color = 'tab:blue'
        ax1.set_xlabel('Elapsed Time (seconds)', fontsize=11)
        ax1.set_ylabel(left_col, color=color, fontsize=11)
        ax1.plot(x_axis_seconds, df_chart['left_val'], color=color, linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        
        if left_y_limits:
            ax1.set_ylim(left_y_limits)
        
        if right_col and right_col != 'None':
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(right_col, color=color, fontsize=11)
            ax2.plot(x_axis_seconds, df_chart['right_val'], color=color, linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor=color)
            
            if right_y_limits:
                ax2.set_ylim(right_y_limits)
        
        if x_limits:
            ax1.set_xlim(x_limits)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def generate_yokogawa_temp_chart(log_data: LogData, x_limits=None, y_limits=None):
        """改進版YOKOGAWA溫度圖表"""
        df = log_data.filter_by_time(x_limits)
        
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(10.2, 5.1))
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        cols_to_plot = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
        
        max_channels = 15
        if len(cols_to_plot) > max_channels:
            cols_to_plot = cols_to_plot[:max_channels]
        
        for col in cols_to_plot:
            y_data = pd.to_numeric(df[col], errors='coerce')
            if not y_data.isna().all():
                display_name = col.replace('YOKO: ', '') if col.startswith('YOKO: ') else col
                ax.plot(df.index.total_seconds(), y_data, label=display_name, linewidth=1)
        
        ax.set_title("YOKOGAWA All Channel Temperature Plot", fontsize=14, fontweight='bold')
        ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=11)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(title="Channels", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
        
        if x_limits:
            ax.set_xlim(x_limits)
        
        if y_limits:
            ax.set_ylim(y_limits)
        
        fig.tight_layout()
        return fig

# =============================================================================
# 7. UI渲染層 (UI Rendering Layer)
# =============================================================================

class GPUMonRenderer:
    """GPUMon UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render_controls(self, file_index=None):
        """渲染控制面板"""
        # 獲取當前檔案索引用於生成唯一key
        if file_index is None:
            file_index = getattr(st.session_state, 'current_file_index', 0)
        key_prefix = f"gpu_{file_index}_"
        
        st.sidebar.markdown("### ⚙️ GPUMon 圖表設定")
        
        numeric_columns = self.log_data.numeric_columns
        if not numeric_columns:
            return None, None, None, None, None
        
        st.sidebar.markdown("#### 🎯 參數選擇")
        
        default_left_index = 0
        for i, col in enumerate(numeric_columns):
            if 'Temperature GPU' in col and '(C)' in col:
                default_left_index = i
                break
        
        left_y_axis = st.sidebar.selectbox(
            "📈 左側Y軸變數", 
            options=numeric_columns, 
            index=default_left_index,
            key=f"{key_prefix}left_y_axis"
        )
        
        right_y_axis_options = ['None'] + numeric_columns
        default_right_index = 0
        for i, col in enumerate(right_y_axis_options):
            if 'TGP' in col and '(W)' in col:
                default_right_index = i
                break
        
        right_y_axis = st.sidebar.selectbox(
            "📊 右側Y軸變數 (可選)", 
            options=right_y_axis_options, 
            index=default_right_index,
            key=f"{key_prefix}right_y_axis"
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
        
        left_y_range_enabled = st.sidebar.checkbox("🔵 啟用左側Y軸範圍限制", key=f"{key_prefix}left_y_range_enabled")
        left_y_range = None
        if left_y_range_enabled:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                left_y_min = st.number_input("左Y軸最小值", value=0.0, key=f"{key_prefix}left_y_min")
            with col2:
                left_y_max = st.number_input("左Y軸最大值", value=100.0, key=f"{key_prefix}left_y_max")
            left_y_range = (left_y_min, left_y_max)
        
        right_y_range = None
        if right_y_axis and right_y_axis != 'None':
            right_y_range_enabled = st.sidebar.checkbox("🔴 啟用右側Y軸範圍限制", key=f"{key_prefix}right_y_range_enabled")
            if right_y_range_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    right_y_min = st.number_input("右Y軸最小值", value=0.0, key=f"{key_prefix}right_y_min")
                with col2:
                    right_y_max = st.number_input("右Y軸最大值", value=100.0, key=f"{key_prefix}right_y_max")
                right_y_range = (right_y_min, right_y_max)
        
        return left_y_axis, right_y_axis, x_range, left_y_range, right_y_range
    
    def render_chart(self, left_col, right_col, x_range, left_y_range, right_y_range):
        """渲染圖表"""
        st.markdown("### 📊 GPUMon 性能監控圖表")
        
        chart = self.chart_gen.generate_gpumon_chart(
            self.log_data, left_col, right_col, x_range, left_y_range, right_y_range
        )
        if chart:
            st.pyplot(chart)
        else:
            st.warning("無法生成圖表，請檢查參數設定")
    
    def render_statistics(self, x_range):
        """渲染統計數據"""
        st.markdown("### 📈 GPUMon 統計數據")
        
        temp_stats, power_stats, freq_stats, util_stats = self.stats_calc.calculate_gpumon_stats(
            self.log_data, x_range
        )
        
        if temp_stats is not None and not temp_stats.empty:
            st.markdown("#### 🌡️ GPU 溫度統計")
            st.dataframe(temp_stats, use_container_width=True, hide_index=True)
        
        if power_stats is not None and not power_stats.empty:
            st.markdown("#### 🔋 GPU 功耗統計")
            st.dataframe(power_stats, use_container_width=True, hide_index=True)
        
        if freq_stats is not None and not freq_stats.empty:
            st.markdown("#### ⚡ GPU 頻率統計")
            st.dataframe(freq_stats, use_container_width=True, hide_index=True)
        
        if util_stats is not None and not util_stats.empty:
            st.markdown("#### 📊 GPU 使用率統計")
            st.dataframe(util_stats, use_container_width=True, hide_index=True)
    
    def render(self, file_index=None):
        """渲染完整UI"""
        st.markdown("""
        <div class="gpumon-box">
            <h4>🎮 GPUMon Log 解析完成！</h4>
            <p>已識別為GPU監控數據，包含溫度、功耗、頻率等指標</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"📊 數據載入：{self.log_data.metadata.rows} 行 × {self.log_data.metadata.columns} 列")
        
        left_col, right_col, x_range, left_y_range, right_y_range = self.render_controls(file_index)
        
        if left_col:
            self.render_chart(left_col, right_col, x_range, left_y_range, right_y_range)
            self.render_statistics(x_range)

class PTATRenderer:
    """PTAT UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render_controls(self, file_index=None):
        """渲染控制面板"""
        # 獲取當前檔案索引用於生成唯一key
        if file_index is None:
            file_index = getattr(st.session_state, 'current_file_index', 0)
        key_prefix = f"ptat_{file_index}_"
        
        st.sidebar.markdown("### ⚙️ PTAT 圖表設定")
        
        numeric_columns = self.log_data.numeric_columns
        if not numeric_columns:
            return None, None, None, None, None
        
        st.sidebar.markdown("#### 🎯 參數選擇")
        
        default_left_index = 0
        for i, col in enumerate(numeric_columns):
            if 'MSR' in col and 'Package' in col and 'Temperature' in col:
                default_left_index = i
                break
        
        left_y_axis = st.sidebar.selectbox("📈 左側Y軸變數", options=numeric_columns, index=default_left_index, key=f"{key_prefix}left_y_axis")
        
        right_y_axis_options = ['None'] + numeric_columns
        default_right_index = 0
        for i, col in enumerate(right_y_axis_options):
            if 'Package' in col and 'Power' in col:
                default_right_index = i
                break
        
        right_y_axis = st.sidebar.selectbox("📊 右側Y軸變數 (可選)", options=right_y_axis_options, index=default_right_index, key=f"{key_prefix}right_y_axis")
        
        st.sidebar.markdown("#### ⏱️ 時間範圍設定")
        
        time_min, time_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider("選擇時間範圍 (秒)", min_value=time_min, max_value=time_max, value=(time_min, time_max), step=1.0, key=f"{key_prefix}x_range")
        
        st.sidebar.markdown("#### 📏 Y軸範圍設定")
        
        left_y_range_enabled = st.sidebar.checkbox("🔵 啟用左側Y軸範圍限制", key=f"{key_prefix}left_y_range_enabled")
        left_y_range = None
        if left_y_range_enabled:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                left_y_min = st.number_input("左Y軸最小值", value=0.0, key=f"{key_prefix}left_y_min")
            with col2:
                left_y_max = st.number_input("左Y軸最大值", value=100.0, key=f"{key_prefix}left_y_max")
            left_y_range = (left_y_min, left_y_max)
        
        right_y_range = None
        if right_y_axis and right_y_axis != 'None':
            right_y_range_enabled = st.sidebar.checkbox("🔴 啟用右側Y軸範圍限制", key=f"{key_prefix}right_y_range_enabled")
            if right_y_range_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    right_y_min = st.number_input("右Y軸最小值", value=0.0, key=f"{key_prefix}right_y_min")
                with col2:
                    right_y_max = st.number_input("右Y軸最大值", value=100.0, key=f"{key_prefix}right_y_max")
                right_y_range = (right_y_min, right_y_max)
        
        return left_y_axis, right_y_axis, x_range, left_y_range, right_y_range
    
    def render(self, file_index=None):
        """渲染完整UI"""
        st.markdown("""
        <div class="info-box">
            <h4>🖥️ PTAT Log 解析完成！</h4>
            <p>已識別為CPU性能監控數據，包含頻率、功耗、溫度等指標</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"📊 數據載入：{self.log_data.metadata.rows} 行 × {self.log_data.metadata.columns} 列")
        
        left_y_axis, right_y_axis, x_range, left_y_range, right_y_range = self.render_controls(file_index)
        
        if left_y_axis:
            st.markdown("### 📊 PTAT CPU 性能監控圖表")
            chart = self.chart_gen.generate_flexible_chart(self.log_data, left_y_axis, right_y_axis, x_range, left_y_range, right_y_range)
            if chart:
                st.pyplot(chart)
            
            st.markdown("### 📈 PTAT 統計數據")
            freq_stats, power_stats, temp_stats = self.stats_calc.calculate_ptat_stats(self.log_data, x_range)
            
            if freq_stats is not None and not freq_stats.empty:
                st.markdown("#### ⚡ CPU 頻率統計")
                st.dataframe(freq_stats, use_container_width=True, hide_index=True)
            
            if power_stats is not None and not power_stats.empty:
                st.markdown("#### 🔋 Package 功耗統計")
                st.dataframe(power_stats, use_container_width=True, hide_index=True)
            
            if temp_stats is not None and not temp_stats.empty:
                st.markdown("#### 🌡️ Package 溫度統計")
                st.dataframe(temp_stats, use_container_width=True, hide_index=True)

class YokogawaRenderer:
    """YOKOGAWA UI渲染器 - v10.3.8 超簡潔版"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render(self, file_index=None):
        """渲染完整UI"""
        # 獲取當前檔案索引用於生成唯一key
        if file_index is None:
            file_index = getattr(st.session_state, 'current_file_index', 0)
        key_prefix = f"yoko_{file_index}_"
        
        st.markdown("""
        <div class="success-box">
            <h4>📊 YOKOGAWA Log 解析完成！ (v10.3.8 多檔案獨立分析版)</h4>
            <p>✨ 智能解析成功，界面清爽，詳細日誌已隱藏在下拉選單中</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"📊 數據載入：{self.log_data.metadata.rows} 行 × {self.log_data.metadata.columns} 列")
        
        st.sidebar.markdown("### ⚙️ YOKOGAWA 圖表設定")
        chart_mode = st.sidebar.radio("📈 圖表模式", ["全通道溫度圖", "自定義雙軸圖"], key=f"{key_prefix}chart_mode")
        
        time_min, time_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider("選擇時間範圍 (秒)", min_value=time_min, max_value=time_max, value=(time_min, time_max), step=1.0, key=f"{key_prefix}x_range")
        
        if chart_mode == "全通道溫度圖":
            st.sidebar.markdown("#### 📏 Y軸範圍設定")
            y_range_enabled = st.sidebar.checkbox("啟用Y軸範圍限制", key=f"{key_prefix}y_range_enabled")
            y_range = None
            if y_range_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    y_min = st.number_input("Y軸最小值", value=0.0, key=f"{key_prefix}y_min")
                with col2:
                    y_max = st.number_input("Y軸最大值", value=100.0, key=f"{key_prefix}y_max")
                y_range = (y_min, y_max)
            
            st.markdown("### 📊 YOKOGAWA 全通道溫度圖表")
            chart = self.chart_gen.generate_yokogawa_temp_chart(self.log_data, x_range, y_range)
            if chart:
                st.pyplot(chart)
        
        else:
            numeric_columns = self.log_data.numeric_columns
            if numeric_columns:
                st.sidebar.markdown("#### 🎯 參數選擇")
                left_y_axis = st.sidebar.selectbox("📈 左側Y軸變數", options=numeric_columns, index=0, key=f"{key_prefix}left_y_axis")
                right_y_axis_options = ['None'] + numeric_columns
                right_y_axis = st.sidebar.selectbox("📊 右側Y軸變數 (可選)", options=right_y_axis_options, index=0, key=f"{key_prefix}right_y_axis")
                
                st.sidebar.markdown("#### 📏 Y軸範圍設定")
                
                left_y_range_enabled = st.sidebar.checkbox("🔵 啟用左側Y軸範圍限制", key=f"{key_prefix}left_y_range_enabled")
                left_y_range = None
                if left_y_range_enabled:
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        left_y_min = st.number_input("左Y軸最小值", value=0.0, key=f"{key_prefix}left_y_min")
                    with col2:
                        left_y_max = st.number_input("左Y軸最大值", value=100.0, key=f"{key_prefix}left_y_max")
                    left_y_range = (left_y_min, left_y_max)
                
                right_y_range = None
                if right_y_axis and right_y_axis != 'None':
                    right_y_range_enabled = st.sidebar.checkbox("🔴 啟用右側Y軸範圍限制", key=f"{key_prefix}right_y_range_enabled")
                    if right_y_range_enabled:
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            right_y_min = st.number_input("右Y軸最小值", value=0.0, key=f"{key_prefix}right_y_min")
                        with col2:
                            right_y_max = st.number_input("右Y軸最大值", value=100.0, key=f"{key_prefix}right_y_max")
                        right_y_range = (right_y_min, right_y_max)
                
                st.markdown("### 📊 YOKOGAWA 自定義圖表")
                chart = self.chart_gen.generate_flexible_chart(self.log_data, left_y_axis, right_y_axis, x_range, left_y_range, right_y_range)
                if chart:
                    st.pyplot(chart)
        
        st.markdown("### 📈 溫度統計數據")
        temp_stats = self.stats_calc.calculate_temp_stats(self.log_data, x_range)
        if not temp_stats.empty:
            st.dataframe(temp_stats, use_container_width=True, hide_index=True)

class SummaryRenderer:
    """Summary UI渲染器 - v10.3.8簡化版 (僅保留HTML帶表格框的數據呈現)"""
    
    def __init__(self, log_data_list: List[LogData]):
        self.log_data_list = log_data_list
        self.summary_gen = TemperatureSummaryGenerator()
    
    def render(self):
        """渲染Summary標籤頁內容 - 簡化版"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; color: white;">
            <h3>📋 溫度整合摘要報告</h3>
            <p>🎯 整合所有檔案的溫度數據，按照標準格式顯示最高溫度</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 生成摘要表格
        summary_df = self.summary_gen.generate_summary_table(self.log_data_list)
        
        if summary_df.empty:
            st.warning("⚠️ 沒有找到可用的溫度數據")
            return
        
        # 顯示檔案來源信息
        stats = self.summary_gen.get_summary_statistics(summary_df)
        if 'log_types' in stats and stats['log_types']:
            with st.expander("📂 檔案來源詳情", expanded=False):
                unique_files = summary_df['Source File'].unique() if 'Source File' in summary_df.columns else []
                for i, filename in enumerate(unique_files, 1):
                    file_data = summary_df[summary_df['Source File'] == filename] if 'Source File' in summary_df.columns else pd.DataFrame()
                    if not file_data.empty:
                        log_type = file_data['Log Type'].iloc[0] if 'Log Type' in file_data.columns else 'Unknown'
                        channel_count = len(file_data)
                        
                        # 添加類型emoji
                        if "GPUMon" in log_type:
                            emoji = "🎮"
                        elif "PTAT" in log_type:
                            emoji = "🖥️"
                        elif "YOKOGAWA" in log_type:
                            emoji = "📊"
                        else:
                            emoji = "📄"
                        
                        st.write(f"**{i}.** {emoji} `{filename}` ({log_type}) - {channel_count} 個監控點")
        
        # 顯示整合表格
        st.markdown("### 📋 溫度監控點整合表格")
        
        # 格式化顯示表格
        display_df = self.summary_gen.format_summary_table_for_display(summary_df)
        
        if not display_df.empty:
            # 準備HTML表格
            html_table = self._prepare_html_table(display_df)
            
            # HTML表格預覽（預設開啟）
            with st.expander("🔍 HTML表格預覽（可直接複製）", expanded=True):
                st.markdown("**以下是帶邊框的HTML表格，可直接選中複製：**")
                st.markdown(html_table, unsafe_allow_html=True)
                st.info("💡 提示：在上方表格上按住滑鼠左鍵拖拽選中整個表格，然後Ctrl+C複製，到Word中Ctrl+V貼上")
        
        else:
            st.error("❌ 無法生成摘要表格")
    
    def _prepare_html_table(self, display_df: pd.DataFrame) -> str:
        """準備帶邊框的HTML表格格式"""
        if display_df.empty:
            return ""
        
        # 創建HTML表格
        html_parts = []
        
        # 添加CSS樣式
        html_parts.append("""
        <style>
        .temp-table {
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }
        .temp-table th, .temp-table td {
            border: 1px solid #333333;
            padding: 8px;
            text-align: center;
            vertical-align: middle;
        }
        .temp-table th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: #333333;
        }
        .temp-table td {
            background-color: #ffffff;
        }
        .temp-table tr:nth-child(even) td {
            background-color: #f9f9f9;
        }
        </style>
        """)
        
        # 開始表格
        html_parts.append('<table class="temp-table">')
        
        # 表格標題行
        html_parts.append('<thead>')
        html_parts.append('<tr>')
        for header in display_df.columns:
            html_parts.append(f'<th>{header}</th>')
        html_parts.append('</tr>')
        html_parts.append('</thead>')
        
        # 表格數據行
        html_parts.append('<tbody>')
        for _, row in display_df.iterrows():
            html_parts.append('<tr>')
            for value in row:
                # 處理空值
                cell_value = str(value) if pd.notna(value) else ""
                html_parts.append(f'<td>{cell_value}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        
        # 結束表格
        html_parts.append('</table>')
        
        return "\n".join(html_parts)

# =============================================================================
# 8. UI工廠 (UI Factory)
# =============================================================================

class RendererFactory:
    """UI渲染器工廠"""
    
    @staticmethod
    def create_renderer(log_data: LogData):
        """根據log類型創建對應的渲染器"""
        log_type = log_data.metadata.log_type
        
        if log_type == "GPUMon Log":
            return GPUMonRenderer(log_data)
        elif log_type == "PTAT Log":
            return PTATRenderer(log_data)
        elif log_type == "YOKOGAWA Log":
            return YokogawaRenderer(log_data)
        else:
            return None

# =============================================================================
# 9. 主應用程式 (Main Application) - v10.3.8 多檔案獨立分析 + Summary整合版 (簡化版)
# =============================================================================

def display_version_info():
    """顯示版本資訊"""
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"""
        **當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**
        
        ### ✨ 主要功能
        
        - **🎮 GPUMon Log** - GPU性能監控數據解析與視覺化
        - **🖥️ PTAT Log** - CPU性能監控數據解析與視覺化  
        - **📊 YOKOGAWA Log** - 多通道溫度記錄儀數據解析與視覺化
        - **📋 Summary整合** - 多檔案溫度數據整合，生成帶邊框HTML表格
        - **📈 獨立分析** - 每個檔案都有專屬的圖表控制和統計分析
        
        ### 🎯 核心特色
        
        - **智能解析** - 自動識別不同類型的Log檔案格式
        - **多檔案支援** - 同時處理多個檔案，獨立分析
        - **帶邊框表格** - Summary頁面提供可直接複製到Word的HTML表格
        - **即時互動** - 時間範圍和參數調整即時更新圖表數據
        """)


def main():
    """主程式 - v10.3.8 Multi-File Analysis with Summary (Simplified)"""
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
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .gpumon-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .stMetric {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid #dee2e6;
        }
        .temp-summary-table {
            font-size: 0.9em;
        }
        .temp-summary-table th {
            background-color: #f0f2f6;
            font-weight: bold;
            text-align: center;
        }
        .temp-summary-table td {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 標題
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 溫度數據視覺化平台</h1>
        <p>智能解析 YOKOGAWA、PTAT、GPUMon Log 文件 | 多檔案獨立分析 + Summary整合 (簡化版)</p>
        <p><strong>{VERSION}</strong> | {VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    display_version_info()
    
    # 初始化解析器註冊系統
    parser_registry = ParserRegistry()
    parser_registry.register(GPUMonParser())
    parser_registry.register(PTATParser())
    parser_registry.register(YokogawaParser())  # 兜底解析器
    
    # 側邊欄
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True,
        help="v10.3.8 簡化版：多檔案獨立分析 + Summary整合，專注帶邊框表格複製"
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
        
        # 解析檔案 - 超簡潔版本
        log_data_list = []
        for uploaded_file in uploaded_files:
            log_data = parser_registry.parse_file(uploaded_file)
            if log_data:
                log_data_list.append(log_data)
        
        if not log_data_list:
            st.error("❌ 無法解析任何檔案")
            return
        
        # 根據檔案數量決定UI模式
        if len(log_data_list) == 1:
            # 單檔案模式
            log_data = log_data_list[0]
            renderer = RendererFactory.create_renderer(log_data)
            
            if renderer:
                renderer.render(file_index=0)
            else:
                st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
        
        else:
            # 多檔案模式 - 每個檔案獨立顯示 + Summary整合
            st.success(f"📊 多檔案分析模式：成功解析 {len(log_data_list)} 個檔案")
            
            # 創建標籤頁，每個檔案一個標籤 + Summary標籤
            tab_names = []
            
            # 首先添加Summary標籤
            tab_names.append("📋 Summary")
            
            # 然後添加各個檔案的標籤
            for i, log_data in enumerate(log_data_list):
                # 生成標籤名稱
                filename = log_data.metadata.filename
                log_type = log_data.metadata.log_type
                
                # 縮短檔案名稱以適應標籤顯示
                short_name = filename
                if len(filename) > 15:
                    name_parts = filename.split('.')
                    if len(name_parts) > 1:
                        short_name = name_parts[0][:12] + "..." + name_parts[-1]
                    else:
                        short_name = filename[:12] + "..."
                
                # 添加類型emoji
                if "GPUMon" in log_type:
                    tab_name = f"🎮 {short_name}"
                elif "PTAT" in log_type:
                    tab_name = f"🖥️ {short_name}"
                elif "YOKOGAWA" in log_type:
                    tab_name = f"📊 {short_name}"
                else:
                    tab_name = f"📄 {short_name}"
                
                tab_names.append(tab_name)
            
            # 創建標籤頁
            tabs = st.tabs(tab_names)
            
            # 首先渲染Summary標籤頁
            with tabs[0]:
                summary_renderer = SummaryRenderer(log_data_list)
                summary_renderer.render()
            
            # 然後為每個檔案渲染獨立的內容
            for i, (tab, log_data) in enumerate(zip(tabs[1:], log_data_list)):
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
                    
                    # 為每個檔案創建獨立的渲染器
                    renderer = RendererFactory.create_renderer(log_data)
                    
                    if renderer:
                        # 渲染該檔案的完整UI，傳遞正確的file_index
                        renderer.render(file_index=i)
                        
                    else:
                        st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
                        
                        # 顯示基本信息作為備用
                        st.markdown("### 📊 基本數據預覽")
                        if not log_data.df.empty:
                            st.write("**欄位列表：**")
                            for col in log_data.df.columns:
                                st.write(f"- {col}")
                            
                            st.write("**數據樣本（前5行）：**")
                            st.dataframe(log_data.df.head(), use_container_width=True)
            
            # 在標籤頁外提供檔案選擇器（用於側邊欄控制）
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎛️ 多檔案控制")
            
            selected_file_index = st.sidebar.selectbox(
                "選擇要控制的檔案",
                options=range(len(log_data_list)),
                format_func=lambda x: f"{log_data_list[x].metadata.filename} ({log_data_list[x].metadata.log_type})",
                help="選擇要在側邊欄中控制的檔案"
            )
            
            st.sidebar.info(f"💡 當前選擇：{log_data_list[selected_file_index].metadata.filename}")
            # 注意：這個選擇器主要用於顯示信息，實際的控制是在各個tab中獨立進行的
    
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件進行分析")
        
        st.markdown("""
        ### 📋 支援的檔案格式
        
        - **🎮 GPUMon CSV** - GPU性能監控數據（溫度、功耗、頻率、使用率）
        - **🖥️ PTAT CSV** - CPU性能監控數據（頻率、功耗、溫度）
        - **📊 YOKOGAWA Excel/CSV** - 多通道溫度記錄儀數據
        
        ### ✨ 主要功能
        
        - **📋 智能解析** - 自動識別不同類型的Log檔案格式
        - **🎯 多檔案分析** - 同時上傳多個檔案，每個檔案獨立分析
        - **📊 即時互動** - 時間範圍和參數調整即時更新圖表
        - **📋 Summary整合** - 所有溫度數據整合成帶邊框HTML表格
        - **💾 一鍵複製** - HTML表格可直接複製到Word保留格式
        
        ### 🎯 使用流程
        
        1. **上傳檔案** - 在左側選擇一個或多個Log檔案
        2. **查看分析** - 每個檔案都有專屬的標籤頁和圖表控制
        3. **整合報告** - 在Summary標籤頁查看所有溫度數據整合表格
        4. **複製使用** - 直接複製HTML表格到Word或Excel
        """)


if __name__ == "__main__":
    main()

