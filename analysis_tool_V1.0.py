# universal_analysis_platform_v5_final.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re

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

# --- 子模組：YOKOGAWA Log 解析器 (最終修正版) ---
def parse_yokogawa(file_content, is_excel=False):
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        df = read_func(file_content, header=28, thousands=',')
        df.columns = df.columns.str.strip()
        
        # 關鍵修正：使用 'TIME' 作為時間欄位，不再使用 'RT'
        time_column = 'TIME'
        if time_column not in df.columns: return None, f"YOKOGAWA Log中找不到 '{time_column}' 欄位"
        
        # 處理 'HH:MM:SS' 或 'HH:MM:SS.ms' 格式的時間字串
        time_series = pd.to_datetime(df[time_column].astype(str), errors='coerce').dt.time
        df['time_index'] = pd.to_timedelta(time_series.astype(str))

        df.dropna(subset=['time_index'], inplace=True)
        start_time = df['time_index'].iloc[0]
        df['time_index'] = df['time_index'] - start_time

        df = df.add_prefix('YOKO: ')
        df.rename(columns={'YOKO: time_index': 'time_index'}, inplace=True)
        return df.set_index('time_index'), "YOKOGAWA Log"
    except Exception as e:
        return None, f"解析YOKOGAWA Log時出錯: {e}"

# --- 主模組：解析器調度中心 ---
def parse_dispatcher(uploaded_file):
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower()
    
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        df_sniff_yoko = read_func(file_content, header=None, skiprows=27, nrows=1)
        file_content.seek(0)
        for cell in df_sniff_yoko.iloc[0]:
            if isinstance(cell, str) and re.match(r'CH\d{3}', cell.strip()):
                return parse_yokogawa(file_content, is_excel)

        if not is_excel:
            first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(5)])
            file_content.seek(0)
            if 'MSR Package Temperature' in first_lines_text:
                return parse_ptat(file_content)
    except Exception as e:
        return None, f"檔案嗅探失敗: {e}"
            
    return None, "未知的Log檔案格式"

# --- 圖表繪製函式 ---
def generate_flexible_chart(df, left_col, right_col, x_limits):
    if df is None or not left_col or left_col not in df.columns: return None
    if right_col and right_col != 'None' and right_col not in df.columns: return None
    
    df_chart = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s'); x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col and right_col != 'None':
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(12, 6)); plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=16)
    x_axis_seconds = df_chart.index.total_seconds()
    color = 'tab:blue'; ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12); ax1.set_ylabel(left_col, color=color, fontsize=12)
    ax1.plot(x_axis_seconds, df_chart['left_val'], color=color); ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', linewidth=0.5)

    if right_col and right_col != 'None':
        ax2 = ax1.twinx(); color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=12); ax2.plot(x_axis_seconds, df_chart['right_val'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    if x_limits: ax1.set_xlim(x_limits)
    fig.tight_layout()
    return fig

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide"); st.title("通用數據分析平台")
st.sidebar.header("控制面板")
uploaded_files = st.sidebar.file_uploader("上傳Log File (可多選)", type=['csv', 'xlsx'], accept_multiple_files=True)
st.sidebar.info("支援 PTAT Log (*.csv) 和 YOKOGAWA MV1000/2000 (*.csv, *.xlsx) 兩種日誌格式。")

if uploaded_files:
    all_dfs = []; log_types_detected = []
    with st.spinner('正在解析所有Log檔案...'):
        for file in uploaded_files:
            df, log_type = parse_dispatcher(file)
            if df is not None:
                all_dfs.append(df); log_types_detected.append(f"{file.name} (辨識為: {log_type})")
            else:
                st.error(f"檔案 '{file.name}' 解析失敗: {log_type}")

    if all_dfs:
        st.sidebar.success("檔案解析完成！")
        for name in log_types_detected: st.sidebar.markdown(f"- `{name}`")

        master_df = pd.concat(all_dfs); 
        master_df_resampled = master_df.select_dtypes(include=['number']).resample('1S').mean(numeric_only=True).interpolate(method='linear')
        
        numeric_columns = master_df_resampled.columns.tolist()
        if numeric_columns:
            st.sidebar.header("圖表設定")
            default_left_list = [c for c in numeric_columns if 'Temp' in c or 'T_' in c]
            default_left = default_left_list[0] if default_left_list else numeric_columns[0]
            left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, index=numeric_columns.index(default_left) if default_left in numeric_columns else 0)
            
            right_y_axis_options = ['None'] + numeric_columns
            default_right_index = 0
            if len(numeric_columns) > 1:
                default_right_list = [c for c in numeric_columns if 'Power' in c or 'Watt' in c or 'P_' in c]
                default_right = default_right_list[0] if default_right_list else 'None'
                try: default_right_index = right_y_axis_options.index(default_right)
                except ValueError: default_right_index = 1
            right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=default_right_index)
            
            st.sidebar.header("X軸範圍設定 (秒)")
            x_min_val = master_df_resampled.index.min().total_seconds(); x_max_val = master_df_resampled.index.max().total_seconds()
            x_min, x_max = st.sidebar.slider("選擇時間範圍", x_min_val, x_max_val, (x_min_val, x_max_val))
            
            st.header("動態比較圖表")
            fig = generate_flexible_chart(master_df_resampled, left_y_axis, right_y_axis, (x_min, x_max))
            if fig: st.pyplot(fig)
        else:
            st.warning("所有檔案解析後，無可用的數值型數據進行繪圖。")
else:
    st.sidebar.info("請上傳您的 Log File(s) 開始分析。")
