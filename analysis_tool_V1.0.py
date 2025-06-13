# universal_analysis_platform_v3_final.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- 子模組：PTAT Log 解析器 ---
def parse_ptat(file_content):
    try:
        df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
        df.columns = df.columns.str.strip()
        time_column = 'Time'
        if time_column not in df.columns: return None, f"PTAT Log中找不到 '{time_column}' 欄位"

        time_series = df[time_column].astype(str).str.strip()
        time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
        datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
        
        valid_times_mask = datetime_series.notna()
        df = df[valid_times_mask].copy()
        if df.empty: return None, "PTAT Log時間格式無法解析"

        valid_datetimes = datetime_series[valid_times_mask]
        df['time_index'] = valid_datetimes - valid_datetimes.iloc[0]
        
        df = df.add_prefix('PTAT: ')
        df.rename(columns={'PTAT: time_index': 'time_index'}, inplace=True)
        return df.set_index('time_index'), "PTAT Log"
    except Exception as e:
        return None, f"解析PTAT Log時出錯: {e}"

# --- 子模組：YOKOGAWA Log 解析器 ---
def parse_yokogawa(file_content, is_excel=False):
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        df = read_func(file_content, header=28, thousands=',', low_memory=False) # 第29行是標題
        df.columns = df.columns.str.strip()

        time_column = 'RT'
        if time_column not in df.columns: return None, f"YOKOGAWA Log中找不到 '{time_column}' 欄位"
        
        df['time_index'] = pd.to_timedelta(pd.to_numeric(df[time_column], errors='coerce'), unit='s')
        df.dropna(subset=['time_index'], inplace=True)
        
        start_time = df['time_index'].iloc[0]
        df['time_index'] = df['time_index'] - start_time

        df = df.add_prefix('YOKO: ')
        df.rename(columns={'YOKO: time_index': 'time_index'}, inplace=True)
        return df.set_index('time_index'), "YOKOGAWA Log"
    except Exception as e:
        return None, f"解析YOKOGAWA Log時出錯: {e}"

# --- 主模組：解析器調度中心 (智慧辨識版) ---
def parse_dispatcher(uploaded_file):
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower()
    
    try:
        # 方法一：基於 Device Type 的YOKOGAWA檔案嗅探 (最可靠)
        read_func = pd.read_excel if is_excel else pd.read_csv
        # 讀取元數據區域來尋找 Device Type
        df_sniff = read_func(file_content, header=None, skiprows=1, nrows=10) # 讀取前10行元數據
        file_content.seek(0)
        for index, row in df_sniff.iterrows():
            row_str = ' '.join(row.dropna().astype(str))
            if 'Device Type' in row_str and ('MV1000' in row_str or 'MV2000' in row_str):
                return parse_yokogawa(file_content, is_excel)

        # 方法二：如果方法一失敗，則用PTAT的特徵來嗅探
        first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(5)])
        file_content.seek(0)
        if 'MSR Package Temperature' in first_lines_text:
            return parse_ptat(file_content)

    except Exception:
        pass # 如果嗅探過程出錯，就回傳未知
            
    return None, "未知的Log檔案格式"

# --- 圖表繪製函式 ---
def generate_flexible_chart(df, left_col, right_col, x_limits):
    # (此函式內容與前次相同，為求完整一併附上)
    if df is None or not left_col or left_col not in df.columns: return None
    if right_col and right_col != 'None' and right_col not in df.columns: return None
    
    df_chart = df.copy()
    x_min_td, x_max_td = (None, None)
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s'); x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col and right_col != 'None':
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(12, 6)); plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=16)
    x_axis_seconds = df_chart.index.total_seconds()
    color = 'tab:blue'; ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12); ax1.set_ylabel(
