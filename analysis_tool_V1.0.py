# universal_analysis_platform_v7_2_yokogawa_time_range.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re

# --- 子模組：PTAT Log 解析器 (穩定版) ---
def parse_ptat(file_content):
    try:
        df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
        df.columns = df.columns.str.strip()
        time_column = 'Time'
        if time_column not in df.columns: return None, "PTAT Log中找不到 'Time' 欄位"
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

# --- 子模組：YOKOGAWA Log 解析器 (專用修正版) ---
def parse_yokogawa(file_content, is_excel=False):
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        
        # 根據你的文件結構，表頭在第30行（pandas的header=29）
        if is_excel:
            # 首先嘗試第30行作為表頭（這是你文件的實際結構）
            possible_headers = [29, 28, 30, 27]  # 29對應第30行
        else:
            possible_headers = [0, 1, 2]
            
        df = None
        found_time_col = None
        successful_header = None
        
        for header_row in possible_headers:
            try:
                file_content.seek(0)  # 重置文件指針
                df = read_func(file_content, header=header_row, thousands=',')
                
                # 清理欄位名稱
                df.columns = df.columns.str.strip()
                
                # Debug 信息
                st.write(f"嘗試 header={header_row}, 欄位: {list(df.columns[:10])}")  # 只顯示前10個欄位
                
                # 檢查可能的時間欄位名稱
                time_candidates = ['Time', 'TIME', 'time', 'Date', 'DATE', 'date', 
                                 'DateTime', 'DATETIME', 'datetime', '時間', '日期時間',
                                 'Timestamp', 'TIMESTAMP', 'timestamp']
                
                for candidate in time_candidates:
                    if candidate in df.columns:
                        found_time_col = candidate
                        successful_header = header_row
                        break
                
                if found_time_col:
                    st.success(f"成功找到時間欄位 '{found_time_col}' 在 header={header_row}")
                    break
                    
            except Exception as e:
                st.warning(f"Header row {header_row} 失敗: {e}")
                continue
        
        if df is None or found_time_col is None:
            error_msg = f"YOKOGAWA Log中找不到時間欄位。"
            if df is not None:
                error_msg += f" 可用欄位: {list(df.columns)[:15]}"  # 限制顯示欄位數量
            return None, error_msg
        
        time_column = found_time_col
        st.info(f"使用時間欄位: {time_column}, 表頭行: {successful_header}")
        
        # 顯示前幾筆時間數據供debug
        st.write(f"時間欄位前5筆數據: {df[time_column].head().tolist()}")
        
        # 處理時間數據
        time_series = df[time_column].astype(str).str.strip()
        
        # 專門處理 HH:MM:SS 格式（你的文件格式）
        try:
            # 先嘗試直接轉為 timedelta（適用於純時間格式）
            df['time_index'] = pd.to_timedelta(time_series + ':00').fillna(pd.to_timedelta('00:00:00'))
            
            # 檢查是否成功
            if df['time_index'].isna().all():
                raise ValueError("Timedelta 轉換失敗")
                
        except:
            # 備用方案：嘗試 datetime 轉換
            try:
                datetime_series = pd.to_datetime(time_series, format='%H:%M:%S', errors='coerce')
                if datetime_series.notna().sum() == 0:
                    datetime_series = pd.to_datetime(time_series, errors='coerce')
                
                # 轉換為相對時間
                df['time_index'] = datetime_series - datetime_series.iloc[0]
                
            except Exception as e:
                return None, f"無法解析時間格式 '{time_column}': {e}. 樣本: {time_series.head(3).tolist()}"
        
        # 檢查是否有有效的時間數據
        valid_times_mask = df['time_index'].notna()
        if valid_times_mask.sum() == 0:
            return None, f"時間欄位 '{time_column}' 中沒有有效的時間數據"
        
        # 清理無效數據
        df = df[valid_times_mask].copy()
        
        # 確保時間從0開始
        if len(df) > 0:
            start_time = df['time_index'].iloc[0]
            df['time_index'] = df['time_index'] - start_time
        
        # 清理數據：處理 "-OVER" 等非數值
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col != 'time_index':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 添加前綴
        df = df.add_prefix('YOKO: ')
        df.rename(columns={'YOKO: time_index': 'time_index'}, inplace=True)
        
        st.success(f"YOKOGAWA 解析成功！數據行數: {len(df)}, 數值欄位數: {len([c for c in df.columns if df[c].dtype in ['float64', 'int64']])}")
        
        return df.set_index('time_index'), "YOKOGAWA Log"
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        st.error(f"解析YOKOGAWA Log詳細錯誤:\n{error_detail}")
        return None, f"解析YOKOGAWA Log時出錯: {e}"

# --- 主模組：解析器調度中心 (修正版) ---
def parse_dispatcher(uploaded_file):
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
    
    st.info(f"開始解析文件: {filename} ({'Excel' if is_excel else 'CSV'})")
    
    try:
        # 對於Excel文件，直接檢查是否為YOKOGAWA格式
        if is_excel:
            # 檢查第28行是否包含 CH 標識
            try:
                file_content.seek(0)
                df_sniff = pd.read_excel(file_content, header=None, skiprows=27, nrows=1)  # 讀取第28行
                
                ch_found = False
                for _, row in df_sniff.iterrows():
                    for cell in row:
                        if isinstance(cell, str) and re.match(r'CH\d{3}', cell.strip()):
                            ch_found = True
                            break
                    if ch_found:
                        break
                
                if ch_found:
                    st.info("檢測到 YOKOGAWA 格式標識")
                    file_content.seek(0)
                    return parse_yokogawa(file_content, is_excel)
                else:
                    # 即使沒有找到CH標識，也嘗試解析YOKOGAWA（可能格式稍有不同）
                    st.warning("未找到標準 YOKOGAWA CH 標識，但仍嘗試 YOKOGAWA 解析")
                    file_content.seek(0)
                    return parse_yokogawa(file_content, is_excel)
                    
            except Exception as e:
                st.warning(f"Excel 格式檢測失敗: {e}，嘗試直接解析")
                file_content.seek(0)
                return parse_yokogawa(file_content, is_excel)
        
        # CSV文件的PTAT檢查
        else:
            try:
                file_content.seek(0)
                first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(10)])
                file_content.seek(0)
                
                if 'MSR Package Temperature' in first_lines_text:
                    st.info("檢測到 PTAT 格式")
                    return parse_ptat(file_content)
                else:
                    st.warning("未檢測到已知格式，嘗試通用CSV解析")
                    
            except Exception as e:
                st.error(f"CSV 格式檢查失敗: {e}")
        
    except Exception as e:
        st.error(f"文件嗅探失敗: {e}")
        
    return None, f"未知的Log檔案格式: {filename}"

# --- 圖表繪製函式 (修正版) ---
def generate_yokogawa_temp_chart(df, x_limits=None):
    """修正版YOKOGAWA溫度圖表，支援時間範圍調整"""
    if df is None: 
        return None
    
    # 套用時間範圍過濾
    df_chart = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    if df_chart.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    numeric_cols = df_chart.select_dtypes(include=['number']).columns
    cols_to_plot = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
    
    # 限制圖表中的線條數量，避免過於混亂
    max_channels = 15
    if len(cols_to_plot) > max_channels:
        cols_to_plot = cols_to_plot[:max_channels]
        st.warning(f"通道數過多，僅顯示前 {max_channels} 個通道")
    
    for col in cols_to_plot:
        y_data = pd.to_numeric(df_chart[col], errors='coerce')
        # 跳過全為NaN的欄位
        if not y_data.isna().all():
            ax.plot(df_chart.index.total_seconds(), y_data, label=col, linewidth=1)
        
    ax.set_title("YOKOGAWA All Channel Temperature Plot", fontsize=16)
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend(title="Channels", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    
    # 如果有設定X軸範圍，套用到圖表
    if x_limits:
        ax.set_xlim(x_limits)
    
    fig.tight_layout()
    return fig

def generate_flexible_chart(df, left_col, right_col, x_limits):
    if df is None or not left_col or left_col not in df.columns: return None
    if right_col and right_col != 'None' and right_col not in df.columns: return None
    df_chart = df.copy()
    if x_limits:
        x_min_td, x_max_td = pd.to_timedelta(x_limits[0], unit='s'), pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    if df_chart.empty: return None
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
st.set_page_config(layout="wide")
st.title("通用數據分析平台 - YOKOGAWA 時間範圍調整版")
st.sidebar.header("控制面板")
uploaded_files = st.sidebar.file_uploader("上傳Log File (可多選)", type=['csv', 'xlsx'], accept_multiple_files=True)
st.sidebar.info("注意：此版本為YOKOGAWA圖表加入了時間範圍調整功能。")

if uploaded_files:
    # --- 關鍵修正：只在需要時才進行檔案類型預判 ---
    if len(uploaded_files) == 1:
        # 傳遞完整的檔案物件進行預判
        df_check, log_type_check = parse_dispatcher(uploaded_files[0])
        is_single_yokogawa = (log_type_check == "YOKOGAWA Log")
    else:
        is_single_yokogawa = False

    # --- YOKOGAWA 專屬顯示模式 (新增時間範圍調整) ---
    if is_single_yokogawa:
        st.sidebar.success(f"檔案 '{uploaded_files[0].name}'\n(辨識為: YOKOGAWA Log)")
        
        # 🔥 新增：YOKOGAWA 時間範圍調整控制項
        st.sidebar.header("YOKOGAWA 圖表設定")
        if df_check is not None and len(df_check) > 0:
            # 計算時間範圍
            x_min_val = df_check.index.min().total_seconds()
            x_max_val = df_check.index.max().total_seconds()
            
            if x_min_val < x_max_val:
                x_min, x_max = st.sidebar.slider(
                    "選擇時間範圍 (秒)", 
                    float(x_min_val), 
                    float(x_max_val), 
                    (float(x_min_val), float(x_max_val)),
                    key="yokogawa_time_range"
                )
                x_limits = (x_min, x_max)
            else:
                st.sidebar.write("數據時間範圍不足。")
                x_limits = None
        else:
            x_limits = None
        
        st.header("YOKOGAWA 全通道溫度曲線圖")
        
        # 直接使用預判時已解析的 DataFrame，並套用時間範圍
        if df_check is not None:
            st.write(f"數據概況：{len(df_check)} 筆記錄，{len(df_check.columns)} 個欄位")
            
            # 🔥 修正：將時間範圍參數傳遞給圖表函式
            fig = generate_yokogawa_temp_chart(df_check, x_limits)
            if fig: 
                st.pyplot(fig)
            else: 
                st.warning("無法產生YOKOGAWA溫度圖表。")
        else:
            st.error("DataFrame 為空，無法繪製圖表")

    # --- 通用互動式分析模式 (適用於PTAT或多檔案) ---
    else:
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
                default_left_list = [c for c in numeric_columns if 'Temp' in c or 'T_' in c or 'CPU' in c]
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
                if x_min_val < x_max_val:
                    x_min, x_max = st.sidebar.slider("選擇時間範圍", x_min_val, x_max_val, (x_min_val, x_max_val))
                else:
                    st.sidebar.write("數據時間範圍不足。"); x_min, x_max = x_min_val, x_max_val
                st.header("動態比較圖表")
                fig = generate_flexible_chart(master_df_resampled, left_y_axis, right_y_axis, (x_min, x_max))
                if fig: st.pyplot(fig)
            else:
                st.warning("所有檔案解析後，無可用的數值型數據進行繪圖。")
else:
    st.sidebar.info("請上傳您的 Log File(s) 開始分析。")
