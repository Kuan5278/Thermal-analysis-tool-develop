# universal_analysis_platform_v7_3_improved.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re

# --- 子模組：PTAT Log 解析器 (靜默版) ---
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

# --- 子模組：YOKOGAWA Log 解析器 (靜默智能版) ---
def parse_yokogawa(file_content, is_excel=False):
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        
        # 根據你的文件結構，表頭在第30行（pandas的header=29）
        if is_excel:
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
                    break
                    
            except Exception as e:
                continue
        
        if df is None or found_time_col is None:
            error_msg = f"YOKOGAWA Log中找不到時間欄位。"
            if df is not None:
                error_msg += f" 可用欄位: {list(df.columns)[:15]}"
            return None, error_msg
        
        time_column = found_time_col
        
        # 🔥 智能欄位重命名功能（靜默執行）
        if is_excel and successful_header == 29:  # 第30行作為表頭
            try:
                # 讀取第28行(CH編號)和第29行(Tag標籤)
                file_content.seek(0)
                ch_row = pd.read_excel(file_content, header=None, skiprows=27, nrows=1).iloc[0]  # 第28行
                file_content.seek(0)
                tag_row = pd.read_excel(file_content, header=None, skiprows=28, nrows=1).iloc[0]  # 第29行
                
                # 建立新的欄位名稱映射
                new_column_names = {}
                for i, original_col in enumerate(df.columns):
                    if i < len(ch_row) and i < len(tag_row):
                        ch_name = str(ch_row.iloc[i]).strip() if pd.notna(ch_row.iloc[i]) else ""
                        tag_name = str(tag_row.iloc[i]).strip() if pd.notna(tag_row.iloc[i]) else ""
                        
                        # 智能命名邏輯：優先使用Tag，為空則使用CH編號
                        if tag_name and tag_name != 'nan' and tag_name != 'Tag':
                            new_column_names[original_col] = tag_name
                        elif ch_name and ch_name != 'nan' and ch_name.startswith('CH'):
                            new_column_names[original_col] = ch_name
                        else:
                            new_column_names[original_col] = original_col  # 保持原名
                    else:
                        new_column_names[original_col] = original_col
                
                # 套用新的欄位名稱
                df.rename(columns=new_column_names, inplace=True)
                
            except Exception as e:
                pass  # 靜默處理重命名失敗
        
        # 處理時間數據
        time_series = df[time_column].astype(str).str.strip()
        
        # 專門處理 HH:MM:SS 格式
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
        
        return df.set_index('time_index'), "YOKOGAWA Log"
        
    except Exception as e:
        return None, f"解析YOKOGAWA Log時出錯: {e}"

# --- 主模組：解析器調度中心 (靜默版) ---
def parse_dispatcher(uploaded_file):
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
    
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
                
                file_content.seek(0)
                return parse_yokogawa(file_content, is_excel)
                    
            except Exception as e:
                file_content.seek(0)
                return parse_yokogawa(file_content, is_excel)
        
        # CSV文件的PTAT檢查
        else:
            try:
                file_content.seek(0)
                first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(10)])
                file_content.seek(0)
                
                if 'MSR Package Temperature' in first_lines_text:
                    return parse_ptat(file_content)
                else:
                    # 嘗試通用CSV解析
                    pass
                    
            except Exception as e:
                pass
        
    except Exception as e:
        pass
        
    return None, f"未知的Log檔案格式: {filename}"

# --- 溫度統計計算函式 ---
def calculate_temp_stats(df, x_limits=None):
    """計算溫度統計數據（最大值和平均值）"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # 套用時間範圍過濾
    df_stats = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_stats = df_stats[(df_stats.index >= x_min_td) & (df_stats.index <= x_max_td)]
    
    if df_stats.empty:
        return pd.DataFrame()
    
    # 找出數值型欄位（排除時間相關欄位）
    numeric_cols = df_stats.select_dtypes(include=['number']).columns
    temp_cols = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
    
    # 計算統計數據
    stats_data = []
    for col in temp_cols:
        y_data = pd.to_numeric(df_stats[col], errors='coerce')
        if not y_data.isna().all():
            t_max = y_data.max()
            t_avg = y_data.mean()
            stats_data.append({
                '通道名稱': col,
                'Tmax (°C)': f"{t_max:.2f}" if pd.notna(t_max) else "N/A",
                'Tavg (°C)': f"{t_avg:.2f}" if pd.notna(t_avg) else "N/A"
            })
    
    return pd.DataFrame(stats_data)

# --- PTAT Log 專用統計計算函式 ---
def calculate_ptat_stats(df, x_limits=None):
    """計算PTAT Log的專用統計數據"""
    if df is None or df.empty:
        return None, None, None
    
    # 套用時間範圍過濾
    df_stats = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_stats = df_stats[(df_stats.index >= x_min_td) & (df_stats.index <= x_max_td)]
    
    if df_stats.empty:
        return None, None, None
    
    # 1. CPU Core Frequency 統計
    freq_stats = []
    freq_cols = [col for col in df_stats.columns if 'frequency' in col.lower() and 'core' in col.lower()]
    
    # 找出LFM和HFM參考值（通常在某些特定欄位中）
    lfm_value = "N/A"
    hfm_value = "N/A"
    
    # 嘗試從欄位名稱或數據中找到LFM/HFM值
    for col in df_stats.columns:
        if 'lfm' in col.lower():
            lfm_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
            if len(lfm_data) > 0:
                lfm_value = f"{lfm_data.iloc[0]:.0f} MHz"
        elif 'hfm' in col.lower():
            hfm_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
            if len(hfm_data) > 0:
                hfm_value = f"{hfm_data.iloc[0]:.0f} MHz"
    
    # 如果沒有找到專用的LFM/HFM欄位，從頻率數據估算
    if lfm_value == "N/A" or hfm_value == "N/A":
        all_freq_data = []
        for col in freq_cols:
            freq_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
            all_freq_data.extend(freq_data.tolist())
        
        if all_freq_data:
            if lfm_value == "N/A":
                lfm_value = f"{min(all_freq_data):.0f} MHz (估算)"
            if hfm_value == "N/A":
                hfm_value = f"{max(all_freq_data):.0f} MHz (估算)"
    
    for col in freq_cols:
        freq_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(freq_data) > 0:
            freq_stats.append({
                'Core': col.replace('PTAT: ', ''),
                'Max (MHz)': f"{freq_data.max():.0f}",
                'Min (MHz)': f"{freq_data.min():.0f}",
                'Avg (MHz)': f"{freq_data.mean():.0f}"
            })
    
    # 添加LFM/HFM參考值
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
    
    # 2. Package Power 統計
    power_stats = []
    power_cols = [col for col in df_stats.columns if 'power' in col.lower() and 'package' in col.lower()]
    
    for col in power_cols:
        power_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(power_data) > 0:
            power_stats.append({
                'Power Type': col.replace('PTAT: ', ''),
                'Max (W)': f"{power_data.max():.2f}",
                'Min (W)': f"{power_data.min():.2f}",
                'Avg (W)': f"{power_data.mean():.2f}"
            })
    
    power_df = pd.DataFrame(power_stats) if power_stats else None
    
    # 3. MSR Package Temperature 統計
    temp_stats = []
    temp_cols = [col for col in df_stats.columns if 'temperature' in col.lower() and 'package' in col.lower() and 'msr' in col.lower()]
    
    for col in temp_cols:
        temp_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(temp_data) > 0:
            temp_stats.append({
                'Temperature Type': col.replace('PTAT: ', ''),
                'Max (°C)': f"{temp_data.max():.2f}",
                'Min (°C)': f"{temp_data.min():.2f}",
                'Avg (°C)': f"{temp_data.mean():.2f}"
            })
    
    temp_df = pd.DataFrame(temp_stats) if temp_stats else None
    
    return freq_df, power_df, temp_df

# --- 圖表繪製函式 (改進版) ---
def generate_yokogawa_temp_chart(df, x_limits=None, y_limits=None):
    """改進版YOKOGAWA溫度圖表，支援時間範圍和Y軸範圍調整"""
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
    
    # 套用X軸範圍
    if x_limits:
        ax.set_xlim(x_limits)
    
    # 🔥 新增：套用Y軸範圍
    if y_limits:
        ax.set_ylim(y_limits)
    
    fig.tight_layout()
    return fig

def generate_flexible_chart(df, left_col, right_col, x_limits, y_limits=None):
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
    
    # 套用Y軸範圍（左軸）
    if y_limits:
        ax1.set_ylim(y_limits)
    
    if right_col and right_col != 'None':
        ax2 = ax1.twinx(); color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=12); ax2.plot(x_axis_seconds, df_chart['right_val'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
    if x_limits: ax1.set_xlim(x_limits)
    fig.tight_layout()
    return fig

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide")
st.title("通用數據分析平台 - 改進版")
st.sidebar.header("控制面板")
uploaded_files = st.sidebar.file_uploader("上傳Log File (可多選)", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    # 🔥 靜默檢測檔案類型
    if len(uploaded_files) == 1:
        df_check, log_type_check = parse_dispatcher(uploaded_files[0])
        is_single_yokogawa = (log_type_check == "YOKOGAWA Log")
    else:
        is_single_yokogawa = False

    # --- YOKOGAWA 專屬顯示模式 ---
    if is_single_yokogawa:
        st.sidebar.success(f"檔案: {uploaded_files[0].name}")
        
        # YOKOGAWA 圖表設定
        st.sidebar.header("圖表設定")
        if df_check is not None and len(df_check) > 0:
            # 時間範圍設定
            x_min_val = df_check.index.min().total_seconds()
            x_max_val = df_check.index.max().total_seconds()
            
            if x_min_val < x_max_val:
                x_min, x_max = st.sidebar.slider(
                    "時間範圍 (秒)", 
                    float(x_min_val), 
                    float(x_max_val), 
                    (float(x_min_val), float(x_max_val)),
                    key="yokogawa_time_range"
                )
                x_limits = (x_min, x_max)
            else:
                x_limits = None
            
            # 🔥 新增：Y軸溫度範圍設定
            st.sidebar.subheader("Y軸溫度範圍設定")
            
            # 計算當前時間範圍內的溫度範圍
            df_temp = df_check.copy()
            if x_limits:
                x_min_td = pd.to_timedelta(x_limits[0], unit='s')
                x_max_td = pd.to_timedelta(x_limits[1], unit='s')
                df_temp = df_temp[(df_temp.index >= x_min_td) & (df_temp.index <= x_max_td)]
            
            if not df_temp.empty:
                numeric_cols = df_temp.select_dtypes(include=['number']).columns
                temp_cols = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
                
                if temp_cols:
                    # 計算所有溫度通道的最小最大值
                    all_temps = pd.concat([pd.to_numeric(df_temp[col], errors='coerce') for col in temp_cols])
                    all_temps = all_temps.dropna()
                    
                    if len(all_temps) > 0:
                        temp_min = float(all_temps.min())
                        temp_max = float(all_temps.max())
                        
                        # 添加一些緩衝空間
                        temp_range = temp_max - temp_min
                        buffer = temp_range * 0.1 if temp_range > 0 else 5
                        
                        auto_y_range = st.sidebar.checkbox("自動Y軸範圍", value=True)
                        
                        if not auto_y_range:
                            y_min, y_max = st.sidebar.slider(
                                "溫度範圍 (°C)",
                                temp_min - buffer,
                                temp_max + buffer,
                                (temp_min - buffer, temp_max + buffer),
                                step=0.1,
                                key="yokogawa_y_range"
                            )
                            y_limits = (y_min, y_max)
                        else:
                            y_limits = None
                    else:
                        y_limits = None
                else:
                    y_limits = None
            else:
                y_limits = None
        else:
            x_limits = None
            y_limits = None
        
        # 顯示圖表
        st.header("YOKOGAWA 全通道溫度曲線圖")
        
        if df_check is not None:
            # 顯示數據概況（簡化版）
            st.write(f"📊 數據記錄：{len(df_check)} 筆，通道數：{len([c for c in df_check.columns if df_check[c].dtype in ['float64', 'int64']])} 個")
            
            # 生成圖表
            fig = generate_yokogawa_temp_chart(df_check, x_limits, y_limits)
            if fig: 
                st.pyplot(fig)
                
                # 🔥 新增：顯示統計表格
                st.subheader("溫度統計數據")
                stats_df = calculate_temp_stats(df_check, x_limits)
                if not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.write("無統計數據可顯示")
            else: 
                st.warning("無法產生溫度圖表")
        else:
            st.error("數據解析失敗")

    # --- 通用互動式分析模式 ---
    else:
        all_dfs = []; log_types = []
        for file in uploaded_files:
            df, log_type = parse_dispatcher(file)
            if df is not None:
                all_dfs.append(df)
                log_types.append(log_type)
        
        if all_dfs:
            # 檢查是否有PTAT Log
            has_ptat = any("PTAT" in log_type for log_type in log_types)
            
            if has_ptat and len(all_dfs) == 1:
                # 單一PTAT Log的特殊處理
                ptat_df = all_dfs[0]
                
                st.sidebar.header("PTAT 圖表設定")
                
                # 時間範圍設定
                if len(ptat_df) > 0:
                    x_min_val = ptat_df.index.min().total_seconds()
                    x_max_val = ptat_df.index.max().total_seconds()
                    
                    if x_min_val < x_max_val:
                        x_min, x_max = st.sidebar.slider(
                            "時間範圍 (秒)", 
                            float(x_min_val), 
                            float(x_max_val), 
                            (float(x_min_val), float(x_max_val)),
                            key="ptat_time_range"
                        )
                        x_limits = (x_min, x_max)
                    else:
                        x_limits = None
                else:
                    x_limits = None
                
                # 變數選擇
                numeric_columns = ptat_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_columns:
                    default_left_list = [c for c in numeric_columns if 'Temp' in c or 'temperature' in c.lower()]
                    default_left = default_left_list[0] if default_left_list else numeric_columns[0]
                    left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, 
                                                     index=numeric_columns.index(default_left) if default_left in numeric_columns else 0)
                    
                    right_y_axis_options = ['None'] + numeric_columns
                    default_right_list = [c for c in numeric_columns if 'Power' in c or 'power' in c.lower()]
                    default_right = default_right_list[0] if default_right_list else 'None'
                    try: 
                        default_right_index = right_y_axis_options.index(default_right)
                    except ValueError: 
                        default_right_index = 0
                    right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=default_right_index)
                    
                    # Y軸範圍設定
                    auto_y = st.sidebar.checkbox("自動Y軸範圍", value=True)
                    y_limits = None
                    if not auto_y and left_y_axis:
                        left_data = pd.to_numeric(ptat_df[left_y_axis], errors='coerce').dropna()
                        if len(left_data) > 0:
                            data_min, data_max = float(left_data.min()), float(left_data.max())
                            data_range = data_max - data_min
                            buffer = data_range * 0.1 if data_range > 0 else 1
                            y_min, y_max = st.sidebar.slider(
                                f"{left_y_axis} 範圍",
                                data_min - buffer,
                                data_max + buffer,
                                (data_min - buffer, data_max + buffer),
                                step=0.1
                            )
                            y_limits = (y_min, y_max)
                    
                    # 顯示圖表
                    st.header("PTAT Log 數據分析")
                    st.write(f"📊 數據記錄：{len(ptat_df)} 筆，參數數：{len(numeric_columns)} 個")
                    
                    fig = generate_flexible_chart(ptat_df, left_y_axis, right_y_axis, x_limits, y_limits)
                    if fig: 
                        st.pyplot(fig)
                        
                        # 🔥 新增：PTAT Log 專用統計表格
                        st.subheader("PTAT Log 統計分析")
                        
                        freq_df, power_df, temp_df = calculate_ptat_stats(ptat_df, x_limits)
                        
                        # 使用分欄布局顯示三個表格
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**CPU Core Frequency 統計**")
                            if freq_df is not None and not freq_df.empty:
                                st.dataframe(freq_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("未找到CPU頻率數據")
                        
                        with col2:
                            st.write("**Package Power 統計**")
                            if power_df is not None and not power_df.empty:
                                st.dataframe(power_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("未找到Package Power數據")
                        
                        with col3:
                            st.write("**MSR Package Temperature 統計**")
                            if temp_df is not None and not temp_df.empty:
                                st.dataframe(temp_df, use_container_width=True, hide_index=True)
                            else:
                                st.write("未找到MSR Package Temperature數據")
                    else:
                        st.warning("無法產生圖表")
                else:
                    st.warning("無可用的數值型數據")
            
            else:
                # 多檔案混合分析模式
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
                    
                    # X軸和Y軸範圍設定
                    st.sidebar.header("軸範圍設定")
                    x_min_val = master_df_resampled.index.min().total_seconds()
                    x_max_val = master_df_resampled.index.max().total_seconds()
                    if x_min_val < x_max_val:
                        x_min, x_max = st.sidebar.slider("時間範圍 (秒)", x_min_val, x_max_val, (x_min_val, x_max_val))
                    else:
                        x_min, x_max = x_min_val, x_max_val
                    
                    # Y軸範圍設定
                    auto_y = st.sidebar.checkbox("自動Y軸範圍", value=True)
                    y_limits = None
                    if not auto_y and left_y_axis:
                        left_data = pd.to_numeric(master_df_resampled[left_y_axis], errors='coerce').dropna()
                        if len(left_data) > 0:
                            data_min, data_max = float(left_data.min()), float(left_data.max())
                            data_range = data_max - data_min
                            buffer = data_range * 0.1 if data_range > 0 else 1
                            y_min, y_max = st.sidebar.slider(
                                f"{left_y_axis} 範圍",
                                data_min - buffer,
                                data_max + buffer,
                                (data_min - buffer, data_max + buffer),
                                step=0.1
                            )
                            y_limits = (y_min, y_max)
                    
                    st.header("動態比較圖表")
                    fig = generate_flexible_chart(master_df_resampled, left_y_axis, right_y_axis, (x_min, x_max), y_limits)
                    if fig: st.pyplot(fig)
                else:
                    st.warning("無可用的數值型數據進行繪圖")
        else:
            st.error("所有檔案解析失敗")
else:
    st.sidebar.info("請上傳您的 Log File(s) 開始分析")
