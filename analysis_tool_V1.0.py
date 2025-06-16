# thermal_analysis_platform_v10.py
# 溫度數據視覺化平台 - v10.2 雙軸Y範圍控制版

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
VERSION = "v10.2 Dual-Axis Range Control"
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
# 2. 解析器層 (Parser Layer)
# =============================================================================

class LogParser(ABC):
    """解析器抽象基類"""
    
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
    """GPUMon解析器"""
    
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
        """解析GPUMon檔案"""
        try:
            file_content.seek(0)
            content = file_content.read().decode('utf-8', errors='ignore')
            lines = content.split('\n')
            
            st.write(f"🔍 GPUMon Debug: 檔案總行數 {len(lines)}")
            
            # 尋找標題行
            header_row_index = self._find_header_row(lines)
            if header_row_index is None:
                return None
            
            # 解析數據
            df = self._parse_data_rows(lines, header_row_index)
            if df is None:
                return None
            
            # 處理時間
            df = self._process_time_data(df)
            if df is None:
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
            
            st.write(f"🎉 GPUMon解析成功! 最終數據: {result_df.shape}")
            return LogData(result_df, metadata)
            
        except Exception as e:
            st.error(f"❌ GPUMon解析錯誤: {e}")
            return None
    
    def _find_header_row(self, lines: List[str]) -> Optional[int]:
        """尋找標題行"""
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if ('iteration' in line_lower and 'date' in line_lower and 'timestamp' in line_lower):
                st.write(f"✅ 找到標題行在第 {i+1} 行")
                return i
        
        # 備用搜尋
        for i, line in enumerate(lines):
            if line.count(',') > 10 and ('iteration' in line.lower() or 'gpu' in line.lower()):
                st.write(f"📍 備用方式找到可能的標題行在第 {i+1} 行")
                return i
        
        return None
    
    def _parse_data_rows(self, lines: List[str], header_row_index: int) -> Optional[pd.DataFrame]:
        """解析數據行"""
        header_line = lines[header_row_index]
        st.write(f"📋 標題行內容: {header_line[:100]}...")
        
        headers = [h.strip() for h in header_line.split(',')]
        st.write(f"📊 解析到 {len(headers)} 個欄位")
        
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
                            if valid_data_count <= 3:
                                st.write(f"✅ 有效數據行 {valid_data_count}: {row_data[:5]}...")
                except Exception:
                    continue
        
        st.write(f"📈 找到 {len(data_rows)} 行有效數據")
        
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
        st.write(f"🎯 DataFrame創建成功: {df.shape}")
        
        return df
    
    def _process_time_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """處理時間數據"""
        try:
            if 'Date' in df.columns and 'Timestamp' in df.columns:
                st.write("🕐 處理時間格式: Date + Timestamp")
                
                df['Timestamp_fixed'] = df['Timestamp'].str.replace(r':(\d{3})$', r'.\1', regex=True)
                st.write(f"🔧 時間格式修正: {df['Timestamp'].iloc[0]} -> {df['Timestamp_fixed'].iloc[0]}")
                
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp_fixed'], errors='coerce')
                st.write(f"📅 合併後時間: {df['DateTime'].iloc[0]}")
                
            else:
                df['DateTime'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(range(len(df)), unit='s')
            
            valid_datetime_count = df['DateTime'].notna().sum()
            st.write(f"📊 成功解析的時間點: {valid_datetime_count}/{len(df)}")
            
            if valid_datetime_count > 0:
                df['time_index'] = df['DateTime'] - df['DateTime'].iloc[0]
                valid_mask = df['time_index'].notna()
                df = df[valid_mask].copy()
                st.write(f"⏰ 時間解析成功，最終有效數據: {len(df)} 行")
            else:
                df['time_index'] = pd.to_timedelta(range(len(df)), unit='s')
            
            return df
            
        except Exception as e:
            st.write(f"⚠️ 時間解析異常: {e}")
            df['time_index'] = pd.to_timedelta(range(len(df)), unit='s')
            return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """轉換數值型欄位"""
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
        
        st.write(f"🔢 轉換了 {numeric_count} 個數值欄位")
        return df

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
        try:
            file_content.seek(0)
            df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
            df.columns = df.columns.str.strip()
            
            if 'Time' not in df.columns:
                return None
            
            time_series = df['Time'].astype(str).str.strip()
            time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
            datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
            
            valid_times_mask = datetime_series.notna()
            df = df[valid_times_mask].copy()
            
            if df.empty:
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
            
            return LogData(result_df, metadata)
            
        except Exception as e:
            return None

class YokogawaParser(LogParser):
    """YOKOGAWA解析器 - 完整版重命名邏輯"""
    
    @property
    def log_type(self) -> str:
        return "YOKOGAWA Log"
    
    def can_parse(self, file_content: io.BytesIO, filename: str) -> bool:
        # YOKOGAWA作為兜底解析器，總是返回True
        return True
    
    def parse(self, file_content: io.BytesIO, filename: str) -> Optional[LogData]:
        st.write(f"🚀 YOKOGAWA解析器啟動 - 檔案: {filename}")
        
        try:
            is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
            read_func = pd.read_excel if is_excel else pd.read_csv
            
            st.write(f"🔍 檔案類型: {'Excel' if is_excel else 'CSV'}")
            
            if is_excel:
                possible_headers = [29, 28, 30, 27]
            else:
                possible_headers = [0, 1, 2]
                
            df = None
            found_time_col = None
            successful_header = None
            
            st.write(f"📋 嘗試header行: {possible_headers}")
            
            for header_row in possible_headers:
                try:
                    file_content.seek(0)
                    df = read_func(file_content, header=header_row, thousands=',')
                    df.columns = df.columns.str.strip()
                    
                    st.write(f"  嘗試header_row={header_row}, 得到形狀: {df.shape}")
                    st.write(f"  欄位: {list(df.columns)[:8]}...")
                    
                    time_candidates = ['Time', 'TIME', 'time', 'Date', 'DATE', 'date', 
                                     'DateTime', 'DATETIME', 'datetime', '時間', '日期時間',
                                     'Timestamp', 'TIMESTAMP', 'timestamp']
                    
                    for candidate in time_candidates:
                        if candidate in df.columns:
                            found_time_col = candidate
                            successful_header = header_row
                            st.write(f"  ✅ 找到時間欄位: {candidate}")
                            break
                    
                    if found_time_col:
                        break
                        
                except Exception as e:
                    st.write(f"  ❌ header_row={header_row} 失敗: {e}")
                    continue
            
            if df is None or found_time_col is None:
                error_msg = f"❌ 無法找到時間欄位。最後嘗試的欄位: {list(df.columns) if df is not None else '無'}"
                st.write(error_msg)
                return None
            
            time_column = found_time_col
            st.write(f"✅ 成功解析，使用header_row={successful_header}, 時間欄位='{time_column}'")
            st.write(f"📊 DataFrame最終形狀: {df.shape}")
            
            # ★★★ 重命名邏輯 - 在任何其他處理之前執行 ★★★
            if is_excel:
                st.write("=" * 50)
                st.write("🏷️ 開始YOKOGAWA欄位重命名邏輯")
                st.write("=" * 50)
                
                try:
                    # 讀取第28行和第29行
                    ch_row_idx = 27  # 第28行
                    tag_row_idx = 28  # 第29行
                    
                    st.write(f"📖 準備讀取第{ch_row_idx+1}行(CH行)和第{tag_row_idx+1}行(Tag行)")
                    
                    # 讀取CH行
                    file_content.seek(0)
                    ch_row = pd.read_excel(file_content, header=None, skiprows=ch_row_idx, nrows=1).iloc[0]
                    st.write(f"✅ 第28行讀取成功，長度: {len(ch_row)}")
                    st.write(f"📋 第28行前10個: {[str(ch_row.iloc[i]) if i < len(ch_row) else 'N/A' for i in range(10)]}")
                    
                    # 讀取Tag行
                    file_content.seek(0)
                    tag_row = pd.read_excel(file_content, header=None, skiprows=tag_row_idx, nrows=1).iloc[0]
                    st.write(f"✅ 第29行讀取成功，長度: {len(tag_row)}")
                    st.write(f"📋 第29行前10個: {[str(tag_row.iloc[i]) if i < len(tag_row) else 'N/A' for i in range(10)]}")
                    
                    # 執行重命名
                    st.write("🔄 開始重命名處理...")
                    original_columns = list(df.columns)
                    st.write(f"📋 原始欄位: {original_columns[:10]}...")
                    
                    new_column_names = {}
                    rename_log = []
                    
                    for i, original_col in enumerate(df.columns):
                        # 獲取CH和Tag名稱
                        ch_name = ""
                        tag_name = ""
                        
                        if i < len(ch_row):
                            ch_val = ch_row.iloc[i]
                            if pd.notna(ch_val) and str(ch_val).strip() not in ['nan', 'NaN', '', ' ', 'None']:
                                ch_name = str(ch_val).strip()
                        
                        if i < len(tag_row):
                            tag_val = tag_row.iloc[i]
                            if pd.notna(tag_val) and str(tag_val).strip() not in ['nan', 'NaN', 'Tag', '', ' ', 'None']:
                                tag_name = str(tag_val).strip()
                        
                        # 決定最終名稱
                        if tag_name:
                            final_name = tag_name
                            rename_log.append(f"欄位{i+1}: '{original_col}' → Tag'{tag_name}'")
                        elif ch_name and ch_name.startswith('CH'):
                            final_name = ch_name
                            rename_log.append(f"欄位{i+1}: '{original_col}' → CH'{ch_name}'")
                        else:
                            final_name = original_col
                            rename_log.append(f"欄位{i+1}: '{original_col}' → 保持原名")
                        
                        new_column_names[original_col] = final_name
                    
                    # 顯示重命名計劃
                    st.write("📝 重命名計劃:")
                    for log_entry in rename_log[:15]:  # 顯示前15個
                        st.write(f"  {log_entry}")
                    
                    # 執行重命名
                    df.rename(columns=new_column_names, inplace=True)
                    st.write(f"✅ 重命名執行完成！")
                    
                    # 驗證重命名結果
                    renamed_columns = list(df.columns)
                    st.write(f"📋 重命名後欄位: {renamed_columns[:10]}...")
                    
                    # 統計重命名效果
                    actual_changes = [(old, new) for old, new in new_column_names.items() if old != new]
                    st.write(f"📊 實際重命名了 {len(actual_changes)} 個欄位:")
                    for old, new in actual_changes[:8]:
                        st.write(f"  '{old}' → '{new}'")
                    
                except Exception as e:
                    st.write(f"❌ 重命名過程異常: {e}")
                    import traceback
                    st.write("詳細錯誤堆疊:")
                    st.code(traceback.format_exc())
                    st.write("⚠️ 將繼續使用原始欄位名稱")
            
            # 繼續處理時間和其他邏輯...
            st.write("⏰ 開始處理時間數據...")
            time_series = df[time_column].astype(str).str.strip()
            
            try:
                df['time_index'] = pd.to_timedelta(time_series + ':00').fillna(pd.to_timedelta('00:00:00'))
                if df['time_index'].isna().all():
                    raise ValueError("Timedelta 轉換失敗")
            except:
                try:
                    datetime_series = pd.to_datetime(time_series, format='%H:%M:%S', errors='coerce')
                    if datetime_series.notna().sum() == 0:
                        datetime_series = pd.to_datetime(time_series, errors='coerce')
                    df['time_index'] = datetime_series - datetime_series.iloc[0]
                except Exception as e:
                    st.write(f"❌ 時間解析失敗: {e}")
                    return None
            
            valid_times_mask = df['time_index'].notna()
            if valid_times_mask.sum() == 0:
                st.write("❌ 沒有有效的時間數據")
                return None
            
            df = df[valid_times_mask].copy()
            
            if len(df) > 0:
                start_time = df['time_index'].iloc[0]
                df['time_index'] = df['time_index'] - start_time
            
            # 數值轉換
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                if col != 'time_index':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 添加前綴
            st.write("🏷️ 添加YOKO前綴...")
            before_prefix = list(df.columns)[:5]
            df = df.add_prefix('YOKO: ')
            df.rename(columns={'YOKO: time_index': 'time_index'}, inplace=True)
            after_prefix = list(df.columns)[:5]
            
            st.write(f"前綴前: {before_prefix}")
            st.write(f"前綴後: {after_prefix}")
            
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
            
            st.write(f"🎉 YOKOGAWA解析完成！最終形狀: {result_df.shape}")
            st.write(f"🏷️ 最終欄位樣本: {list(result_df.columns)[:8]}...")
            
            return LogData(result_df, metadata)
            
        except Exception as e:
            st.write(f"❌ YOKOGAWA解析器整體異常: {e}")
            import traceback
            st.write("完整錯誤堆疊:")
            st.code(traceback.format_exc())
            return None

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
        is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
        
        st.write(f"🔍 檔案分析: {filename} (Excel: {is_excel})")
        
        for parser in self.parsers:
            try:
                file_content.seek(0)
                if parser.can_parse(file_content, filename):
                    st.write(f"🎯 使用 {parser.log_type} 解析器")
                    file_content.seek(0)
                    result = parser.parse(file_content, filename)
                    if result is not None:
                        return result
            except Exception as e:
                st.write(f"⚠️ {parser.log_type} 解析器失敗: {e}")
                continue
        
        return None

# =============================================================================
# 4. 統計計算層 (Statistics Layer) - 功耗項目優化版本
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
        
        # GPU功耗統計 - 只顯示指定的三個項目
        power_stats = []
        # 指定的功耗項目：NVVDD power, FBVDD power, TGP (W)
        target_power_items = ['NVVDD', 'FBVDD', 'TGP']
        
        for target_item in target_power_items:
            # 尋找包含目標項目的欄位
            matching_cols = [col for col in df.columns if target_item in col and ('Power' in col or 'TGP' in col)]
            
            for col in matching_cols:
                power_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(power_data) > 0:
                    # 清理顯示名稱
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
                    break  # 找到一個就停止，避免重複
        
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
        
        # Package Power 統計 - 只顯示指定的四個項目
        power_stats = []
        # 指定的功耗項目：IA power, GT power, Rest of package power, package power
        target_power_items = [
            ('IA', 'IA Power'),
            ('GT', 'GT Power'), 
            ('Rest of package', 'Rest of Package Power'),
            ('Package', 'Package Power')
        ]
        
        for search_term, display_name in target_power_items:
            # 尋找包含目標項目的欄位
            matching_cols = []
            for col in df.columns:
                col_lower = col.lower()
                search_lower = search_term.lower()
                
                # 更精確的匹配邏輯
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
            
            # 如果找到匹配的欄位，取第一個
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
                
                # 清理顯示名稱 - 移除前綴並顯示更友好的名稱
                display_name = col
                if display_name.startswith('YOKO: '):
                    display_name = display_name.replace('YOKO: ', '')
                elif display_name.startswith('PTAT: '):
                    display_name = display_name.replace('PTAT: ', '')
                elif display_name.startswith('GPU: '):
                    display_name = display_name.replace('GPU: ', '')
                
                # 排除明顯的非溫度欄位
                if display_name.lower() in ['sec', 'time', 'rt', 'date']:
                    continue
                
                stats_data.append({
                    '通道名稱': display_name,
                    'Tmax (°C)': f"{t_max:.2f}" if pd.notna(t_max) else "N/A",
                    'Tavg (°C)': f"{t_avg:.2f}" if pd.notna(t_avg) else "N/A"
                })
        
        return pd.DataFrame(stats_data)

# =============================================================================
# 5. 圖表生成層 (Chart Generation Layer)
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
        
        # 設定左側Y軸範圍
        if left_y_limits:
            ax1.set_ylim(left_y_limits)
        
        if right_col and right_col != 'None':
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel(right_col.replace("GPU: ", ""), color=color, fontsize=11)
            ax2.plot(x_axis_seconds, df_chart['right_val'], color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 設定右側Y軸範圍
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
        
        # 設定左側Y軸範圍
        if left_y_limits:
            ax1.set_ylim(left_y_limits)
        
        if right_col and right_col != 'None':
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel(right_col, color=color, fontsize=11)
            ax2.plot(x_axis_seconds, df_chart['right_val'], color=color, linewidth=1.5)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # 設定右側Y軸範圍
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
                # 移除前綴顯示更友好的名稱
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
# 6. UI渲染層 (UI Rendering Layer)
# =============================================================================

class GPUMonRenderer:
    """GPUMon UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render_controls(self):
        """渲染控制面板"""
        st.sidebar.markdown("### ⚙️ GPUMon 圖表設定")
        
        numeric_columns = self.log_data.numeric_columns
        if not numeric_columns:
            return None, None, None, None, None
        
        st.sidebar.markdown("#### 🎯 參數選擇")
        
        # 尋找預設的左軸變數 (Temperature GPU)
        default_left_index = 0
        for i, col in enumerate(numeric_columns):
            if 'Temperature GPU' in col and '(C)' in col:
                default_left_index = i
                break
        
        left_y_axis = st.sidebar.selectbox(
            "📈 左側Y軸變數", 
            options=numeric_columns, 
            index=default_left_index
        )
        
        # 尋找預設的右軸變數 (TGP)
        right_y_axis_options = ['None'] + numeric_columns
        default_right_index = 0
        for i, col in enumerate(right_y_axis_options):
            if 'TGP' in col and '(W)' in col:
                default_right_index = i
                break
        
        right_y_axis = st.sidebar.selectbox(
            "📊 右側Y軸變數 (可選)", 
            options=right_y_axis_options, 
            index=default_right_index
        )
        
        st.sidebar.markdown("#### ⏱️ 時間範圍設定")
        
        time_min, time_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider(
            "選擇時間範圍 (秒)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
            step=1.0
        )
        
        st.sidebar.markdown("#### 📏 Y軸範圍設定")
        
        # 左側Y軸範圍設定
        left_y_range_enabled = st.sidebar.checkbox("🔵 啟用左側Y軸範圍限制")
        left_y_range = None
        if left_y_range_enabled:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                left_y_min = st.number_input("左Y軸最小值", value=0.0, key="left_y_min")
            with col2:
                left_y_max = st.number_input("左Y軸最大值", value=100.0, key="left_y_max")
            left_y_range = (left_y_min, left_y_max)
        
        # 右側Y軸範圍設定（只有在選擇右軸變數時才顯示）
        right_y_range = None
        if right_y_axis and right_y_axis != 'None':
            right_y_range_enabled = st.sidebar.checkbox("🔴 啟用右側Y軸範圍限制")
            if right_y_range_enabled:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    right_y_min = st.number_input("右Y軸最小值", value=0.0, key="right_y_min")
                with col2:
                    right_y_max = st.number_input("右Y軸最大值", value=100.0, key="right_y_max")
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
        
        # 垂直排列顯示所有統計表格，避免併排放不下的問題
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
    
    def render(self):
        """渲染完整UI"""
        st.markdown("""
        <div class="gpumon-box">
            <h4>🎮 GPUMon Log 成功解析！</h4>
            <p>已識別為GPU監控數據，包含溫度、功耗、頻率等指標</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ 數據載入成功：{self.log_data.metadata.rows} 行 × {self.log_data.metadata.columns} 列")
        
        # 渲染控制面板
        left_col, right_col, x_range, left_y_range, right_y_range = self.render_controls()
        
        if left_col:
            # 渲染圖表
            self.render_chart(left_col, right_col, x_range, left_y_range, right_y_range)
            
            # 渲染統計數據
            self.render_statistics(x_range)

class PTATRenderer:
    """PTAT UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render(self):
        """渲染完整UI"""
        st.markdown("""
        <div class="info-box">
            <h4>🖥️ PTAT Log 成功解析！</h4>
            <p>已識別為CPU性能監控數據，包含頻率、功耗、溫度等指標</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ 數據載入成功：{self.log_data.metadata.rows} 行 × {self.log_data.metadata.columns} 列")
        
        # PTAT專用UI邏輯
        numeric_columns = self.log_data.numeric_columns
        if numeric_columns:
            st.sidebar.markdown("### ⚙️ PTAT 圖表設定")
            
            # 尋找預設的左軸變數 (MSR Package Temperature)
            default_left_index = 0
            for i, col in enumerate(numeric_columns):
                if 'MSR' in col and 'Package' in col and 'Temperature' in col:
                    default_left_index = i
                    break
            
            left_y_axis = st.sidebar.selectbox("📈 左側Y軸變數", options=numeric_columns, index=default_left_index)
            
            # 尋找預設的右軸變數 (Package Power)
            right_y_axis_options = ['None'] + numeric_columns
            default_right_index = 0
            for i, col in enumerate(right_y_axis_options):
                if 'Package' in col and 'Power' in col:
                    default_right_index = i
                    break
            
            right_y_axis = st.sidebar.selectbox("📊 右側Y軸變數 (可選)", options=right_y_axis_options, index=default_right_index)
            
            time_min, time_max = self.log_data.get_time_range()
            x_range = st.sidebar.slider("選擇時間範圍 (秒)", min_value=time_min, max_value=time_max, value=(time_min, time_max), step=1.0)
            
            # 顯示圖表
            st.markdown("### 📊 PTAT CPU 性能監控圖表")
            chart = self.chart_gen.generate_flexible_chart(self.log_data, left_y_axis, right_y_axis, x_range)
            if chart:
                st.pyplot(chart)
            
            # 顯示統計數據 - 垂直排列
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
    """YOKOGAWA UI渲染器"""
    
    def __init__(self, log_data: LogData):
        self.log_data = log_data
        self.stats_calc = StatisticsCalculator()
        self.chart_gen = ChartGenerator()
    
    def render(self):
        """渲染完整UI"""
        st.markdown("""
        <div class="success-box">
            <h4>📊 YOKOGAWA Log 成功解析！</h4>
            <p>已識別為溫度記錄儀數據，支援多通道溫度監控</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ 數據載入成功：{self.log_data.metadata.rows} 行 × {self.log_data.metadata.columns} 列")
        
        # YOKOGAWA專用UI邏輯
        st.sidebar.markdown("### ⚙️ YOKOGAWA 圖表設定")
        chart_mode = st.sidebar.radio("📈 圖表模式", ["全通道溫度圖", "自定義雙軸圖"])
        
        time_min, time_max = self.log_data.get_time_range()
        x_range = st.sidebar.slider("選擇時間範圍 (秒)", min_value=time_min, max_value=time_max, value=(time_min, time_max), step=1.0)
        
        if chart_mode == "全通道溫度圖":
            y_range_enabled = st.sidebar.checkbox("啟用Y軸範圍限制")
            y_range = None
            if y_range_enabled:
                y_min = st.sidebar.number_input("Y軸最小值", value=0.0)
                y_max = st.sidebar.number_input("Y軸最大值", value=100.0)
                y_range = (y_min, y_max)
            
            st.markdown("### 📊 YOKOGAWA 全通道溫度圖表")
            chart = self.chart_gen.generate_yokogawa_temp_chart(self.log_data, x_range, y_range)
            if chart:
                st.pyplot(chart)
        
        else:
            numeric_columns = self.log_data.numeric_columns
            if numeric_columns:
                left_y_axis = st.sidebar.selectbox("📈 左側Y軸變數", options=numeric_columns, index=0)
                right_y_axis_options = ['None'] + numeric_columns
                right_y_axis = st.sidebar.selectbox("📊 右側Y軸變數 (可選)", options=right_y_axis_options, index=0)
                
                st.markdown("### 📊 YOKOGAWA 自定義圖表")
                chart = self.chart_gen.generate_flexible_chart(self.log_data, left_y_axis, right_y_axis, x_range)
                if chart:
                    st.pyplot(chart)
        
        # 顯示溫度統計
        st.markdown("### 📈 溫度統計數據")
        temp_stats = self.stats_calc.calculate_temp_stats(self.log_data, x_range)
        if not temp_stats.empty:
            st.dataframe(temp_stats, use_container_width=True, hide_index=True)

# =============================================================================
# 7. UI工廠 (UI Factory)
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
# 8. 主應用程式 (Main Application)
# =============================================================================

def display_version_info():
    """顯示版本資訊"""
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"""
        **當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**
        
        ### 🆕 v10.2 Dual-Axis Range Control 更新內容：
        - 📏 **雙軸Y範圍控制** - 可分別設定左右Y軸的顯示範圍
        - 🔵 **左側Y軸範圍** - 獨立控制左側Y軸的最小值和最大值
        - 🔴 **右側Y軸範圍** - 獨立控制右側Y軸的最小值和最大值（雙軸模式時）
        - 🎯 **智能顯示控制** - 只有選擇右軸變數時才顯示右軸範圍設定
        - 🔋 **功耗統計優化** - 保持 PTAT 4項核心功耗，GPUMon 3項關鍵功耗
        
        ### 🏗️ 架構優勢：
        - 分層架構設計，高擴展性
        - 可插拔解析器系統
        - UI組件化渲染
        - 統一數據模型
        - 智能功耗項目過濾
        
        ---
        💡 **功耗統計焦點：** 現在只顯示最關鍵的功耗指標，讓數據分析更聚焦！
        """)

def main():
    """主程式 - v10.2 Dual-Axis Range Control"""
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
    </style>
    """, unsafe_allow_html=True)
    
    # 標題
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 溫度數據視覺化平台</h1>
        <p>智能解析 YOKOGAWA、PTAT、GPUMon Log 文件，提供數據分析與視覺化</p>
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
        help="支援 YOKOGAWA Excel、PTAT CSV、GPUMon CSV 格式"
    )
    
    # ★★★ 顯示訪問計數器 ★★★
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
        st.markdown("### 🔍 檔案解析調試資訊")
        
        log_data_list = []
        for uploaded_file in uploaded_files:
            log_data = parser_registry.parse_file(uploaded_file)
            if log_data:
                log_data_list.append(log_data)
        
        if not log_data_list:
            st.error("❌ 無法解析任何檔案")
            return
        
        # 根據檔案數量和類型決定UI模式
        if len(log_data_list) == 1:
            log_data = log_data_list[0]
            renderer = RendererFactory.create_renderer(log_data)
            
            if renderer:
                renderer.render()
            else:
                st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
        
        else:
            st.info("📊 多檔案模式，使用基本分析功能")
            
            # 多檔案合併邏輯
            if len(log_data_list) > 1:
                try:
                    combined_df = pd.concat([log_data.df for log_data in log_data_list], axis=1)
                    
                    # 創建合併的LogData
                    combined_metadata = LogMetadata(
                        filename="合併檔案",
                        log_type="混合類型",
                        rows=combined_df.shape[0],
                        columns=combined_df.shape[1],
                        time_range=f"{combined_df.index.min()} 到 {combined_df.index.max()}",
                        file_size_kb=sum(log_data.metadata.file_size_kb for log_data in log_data_list)
                    )
                    
                    combined_log_data = LogData(combined_df, combined_metadata)
                    
                    st.success(f"✅ 合併數據載入成功：{combined_log_data.metadata.rows} 行 × {combined_log_data.metadata.columns} 列")
                    
                    numeric_columns = combined_log_data.numeric_columns
                    if numeric_columns:
                        st.sidebar.markdown("### ⚙️ 圖表設定")
                        left_y_axis = st.sidebar.selectbox("📈 左側Y軸變數", options=numeric_columns, index=0)
                        right_y_axis_options = ['None'] + numeric_columns
                        right_y_axis = st.sidebar.selectbox("📊 右側Y軸變數 (可選)", options=right_y_axis_options, index=0)
                        
                        time_min, time_max = combined_log_data.get_time_range()
                        x_range = st.sidebar.slider("選擇時間範圍 (秒)", min_value=time_min, max_value=time_max, value=(time_min, time_max), step=1.0)
                        
                        st.markdown("### 📊 綜合數據圖表")
                        chart_gen = ChartGenerator()
                        chart = chart_gen.generate_flexible_chart(combined_log_data, left_y_axis, right_y_axis, x_range)
                        if chart:
                            st.pyplot(chart)
                        
                        # 顯示基本統計數據
                        st.markdown("### 📈 基本統計數據")
                        stats_calc = StatisticsCalculator()
                        temp_stats = stats_calc.calculate_temp_stats(combined_log_data, x_range)
                        if not temp_stats.empty:
                            st.dataframe(temp_stats, use_container_width=True, hide_index=True)
                
                except Exception as e:
                    st.error(f"合併數據時出錯: {e}")
            
            else:
                # 單檔案但不是已知類型的情況
                log_data = log_data_list[0]
                st.success(f"✅ 數據載入成功：{log_data.metadata.rows} 行 × {log_data.metadata.columns} 列")
                
                numeric_columns = log_data.numeric_columns
                if numeric_columns:
                    st.sidebar.markdown("### ⚙️ 圖表設定")
                    left_y_axis = st.sidebar.selectbox("📈 左側Y軸變數", options=numeric_columns, index=0)
                    right_y_axis_options = ['None'] + numeric_columns
                    right_y_axis = st.sidebar.selectbox("📊 右側Y軸變數 (可選)", options=right_y_axis_options, index=0)
                    
                    time_min, time_max = log_data.get_time_range()
                    x_range = st.sidebar.slider("選擇時間範圍 (秒)", min_value=time_min, max_value=time_max, value=(time_min, time_max), step=1.0)
                    
                    st.markdown("### 📊 數據圖表")
                    chart_gen = ChartGenerator()
                    chart = chart_gen.generate_flexible_chart(log_data, left_y_axis, right_y_axis, x_range)
                    if chart:
                        st.pyplot(chart)
                    
                    # 顯示基本統計數據
                    st.markdown("### 📈 基本統計數據")
                    stats_calc = StatisticsCalculator()
                    temp_stats = stats_calc.calculate_temp_stats(log_data, x_range)
                    if not temp_stats.empty:
                        st.dataframe(temp_stats, use_container_width=True, hide_index=True)
    
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件進行分析")
        
        st.markdown("""
        ### 📋 支援的檔案格式
        
        - **🎮 GPUMon CSV** - GPU性能監控數據（溫度、功耗、頻率、使用率）
        - **🖥️ PTAT CSV** - CPU性能監控數據（頻率、功耗、溫度）
        - **📊 YOKOGAWA Excel/CSV** - 多通道溫度記錄儀數據
        
        ### 🔍 v10.2 Dual-Axis Range Control 新功能
        
        - **📏 雙軸Y範圍控制** - 可分別設定左右Y軸的顯示範圍
        - **🎯 精確圖表控制** - 左右Y軸獨立調整，數據顯示更精確
        - **🔋 功耗統計優化** - PTAT 4項核心功耗，GPUMon 3項關鍵功耗
        - **🔵🔴 視覺化標示** - 藍色圓點標示左軸，紅色圓點標示右軸
        - **訪問統計** - 持續追蹤平台使用情況和趨勢
        """)

if __name__ == "__main__":
    main()
