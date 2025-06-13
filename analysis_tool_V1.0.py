# universal_analysis_platform_v8_0_enhanced.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime

# --- 版本資訊設定 ---
VERSION = "v1.0"
VERSION_DATE = "2025年6月"
VERSION_FEATURES = [
    "First release",
    
]

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
    
    # 🎨 縮小圖表大小 15%
    fig, ax = plt.subplots(figsize=(10.2, 6.8))  # 原本 (12, 8) 縮小 15%
    
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
        
    ax.set_title("YOKOGAWA All Channel Temperature Plot", fontsize=14, fontweight='bold')
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(title="Channels", bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
    
    # 套用X軸範圍
    if x_limits:
        ax.set_xlim(x_limits)
    
    # 套用Y軸範圍
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
    
    # 🎨 縮小圖表大小 15%
    fig, ax1 = plt.subplots(figsize=(10.2, 5.1))  # 原本 (12, 6) 縮小 15%
    plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=14, fontweight='bold')
    
    x_axis_seconds = df_chart.index.total_seconds()
    color = 'tab:blue'
    ax1.set_xlabel('Elapsed Time (seconds)', fontsize=11)
    ax1.set_ylabel(left_col, color=color, fontsize=11)
    ax1.plot(x_axis_seconds, df_chart['left_val'], color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 套用Y軸範圍（左軸）
    if y_limits:
        ax1.set_ylim(y_limits)
    
    if right_col and right_col != 'None':
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=11)
        ax2.plot(x_axis_seconds, df_chart['right_val'], color=color, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)
    
    if x_limits: 
        ax1.set_xlim(x_limits)
    
    fig.tight_layout()
    return fig

# --- 版本資訊顯示函式 ---
def display_version_info():
    """顯示版本資訊"""
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"""
        **當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**
        
        ### 🆕 本版本更新內容：
        """)
        
        for feature in VERSION_FEATURES:
            st.markdown(f"- {feature}")
        
        st.markdown("---")
        st.markdown("💡 **使用提示：** 支援YOKOGAWA Excel格式、PTAT CSV格式，提供智能解析與多維度統計分析")

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(
    page_title="通用數據分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 自定義CSS樣式
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
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
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 🏠 主頁面標題
st.markdown("""
<div class="main-header">
    <h1>📊 通用數據分析平台</h1>
    <p>智能解析 YOKOGAWA & PTAT Log 文件，提供專業級數據分析與視覺化</p>
</div>
""", unsafe_allow_html=True)

# 🔖 版本資訊區域
display_version_info()

# 📋 側邊欄設計
st.sidebar.markdown("### 🎛️ 控制面板")
st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader(
    "📁 上傳Log File (可多選)", 
    type=['csv', 'xlsx'], 
    accept_multiple_files=True,
    help="支援 YOKOGAWA Excel 格式和 PTAT CSV 格式"
)

if uploaded_files:
    # 📁 檔案資訊顯示
    st.sidebar.markdown("### 📂 已上傳檔案")
    for i, file in enumerate(uploaded_files, 1):
        file_size = len(file.getvalue()) / 1024  # KB
        st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
    
    st.sidebar.markdown("---")
    
    # 🔥 靜默檢測檔案類型
    if len(uploaded_files) == 1:
        df_check, log_type_check = parse_dispatcher(uploaded_files[0])
        is_single_yokogawa = (log_type_check == "YOKOGAWA Log")
    else:
        is_single_yokogawa = False

    # --- YOKOGAWA 專屬顯示模式 ---
    if is_single_yokogawa:
        # 📊 狀態顯示
        st.markdown(f"""
        <div class="success-box">
            <strong>✅ 檔案解析成功</strong><br>
            📄 檔案類型：{log_type_check}<br>
            📊 數據筆數：{len(df_check):,} 筆<br>
            🔢 通道數量：{len([c for c in df_check.columns if df_check[c].dtype in ['float64', 'int64']]):,} 個
        </div>
        """, unsafe_allow_html=True)
        
        # 🎛️ YOKOGAWA 圖表設定
        st.sidebar.markdown("### ⚙️ 圖表設定")
        
        if df_check is not None and len(df_check) > 0:
            # ⏱️ 時間範圍設定
            x_min_val = df_check.index.min().total_seconds()
            x_max_val = df_check.index.max().total_seconds()
            
            if x_min_val < x_max_val:
                x_min, x_max = st.sidebar.slider(
                    "⏱️ 時間範圍 (秒)", 
                    float(x_min_val), 
                    float(x_max_val), 
                    (float(x_min_val), float(x_max_val)),
                    key="yokogawa_time_range"
                )
                x_limits = (x_min, x_max)
            else:
                x_limits = None
            
            # 🎯 Y軸溫度範圍設定
            st.sidebar.markdown("#### 🎯 Y軸溫度範圍")
            
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
                        
                        auto_y_range = st.sidebar.checkbox("🔄 自動Y軸範圍", value=True)
                        
                        if not auto_y_range:
                            y_min, y_max = st.sidebar.slider(
                                "🌡️ 溫度範圍 (°C)",
                                temp_min - buffer,
                                temp_max + buffer,
                                (temp_min - buffer, temp_max + buffer),
                                step=0.1,
                                key="yokogawa_y_range"
                            )
                            y_limits = (y_min, y_max)
                        else:
                            y_limits = None
                            
                        # 📊 溫度範圍資訊顯示
                        st.sidebar.markdown(f"""
                        **📈 當前溫度範圍：**
                        - 最高：{temp_max:.1f}°C
                        - 最低：{temp_min:.1f}°C
                        - 差值：{temp_range:.1f}°C
                        """)
                    else:
                        y_limits = None
                else:
                    y_limits = None
            else:
                y_limits = None
        else:
            x_limits = None
            y_limits = None
        
        # 🏠 主要內容區域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📈 YOKOGAWA 全通道溫度曲線圖")
            
            if df_check is not None:
                # 生成圖表
                fig = generate_yokogawa_temp_chart(df_check, x_limits, y_limits)
                if fig: 
                    st.pyplot(fig)
                else: 
                    st.warning("⚠️ 無法產生溫度圖表")
            else:
                st.error("❌ 數據解析失敗")
        
        with col2:
            st.markdown("### 📊 統計數據")
            stats_df = calculate_temp_stats(df_check, x_limits)
            if not stats_df.empty:
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # 📈 快速統計摘要
                if len(stats_df) > 0:
                    try:
                        max_temps = [float(x.replace('°C', '')) for x in stats_df['Tmax (°C)'] if x != 'N/A']
                        avg_temps = [float(x.replace('°C', '')) for x in stats_df['Tavg (°C)'] if x != 'N/A']
                        
                        if max_temps and avg_temps:
                            st.markdown(f"""
                            <div class="metric-card">
                                <strong>🔥 整體最高溫：</strong> {max(max_temps):.1f}°C<br>
                                <strong>📊 平均溫度：</strong> {sum(avg_temps)/len(avg_temps):.1f}°C<br>
                                <strong>📈 活躍通道：</strong> {len(stats_df)} 個
                            </div>
                            """, unsafe_allow_html=True)
                    except:
                        pass
            else:
                st.markdown("""
                <div class="info-box">
                    ❓ 無統計數據可顯示<br>
                    請檢查時間範圍設定
                </div>
                """, unsafe_allow_html=True)

    # --- 通用互動式分析模式 ---
    else:
        all_dfs = []
        log_types = []
        
        for file in uploaded_files:
            df, log_type = parse_dispatcher(file)
            if df is not None:
                all_dfs.append(df)
                log_types.append(log_type)
        
        if all_dfs:
            # 📊 檔案解析狀態
            st.markdown("### 📋 檔案解析狀態")
            status_cols = st.columns(len(uploaded_files))
            
            for i, (file, log_type) in enumerate(zip(uploaded_files, log_types)):
                with status_cols[i]:
                    if i < len(all_dfs):
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>✅ {file.name}</strong><br>
                            📄 {log_type}<br>
                            📊 {len(all_dfs[i]):,} 筆數據
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 1rem; border-radius: 8px;">
                            <strong>❌ {file.name}</strong><br>
                            解析失敗
                        </div>
                        """, unsafe_allow_html=True)
            
            # 檢查是否有PTAT Log
            has_ptat = any("PTAT" in log_type for log_type in log_types)
            
            if has_ptat and len(all_dfs) == 1:
                # 🔬 單一PTAT Log的特殊處理
                ptat_df = all_dfs[0]
                
                st.sidebar.markdown("### ⚙️ PTAT 圖表設定")
                
                # ⏱️ 時間範圍設定
                if len(ptat_df) > 0:
                    x_min_val = ptat_df.index.min().total_seconds()
                    x_max_val = ptat_df.index.max().total_seconds()
                    
                    if x_min_val < x_max_val:
                        x_min, x_max = st.sidebar.slider(
                            "⏱️ 時間範圍 (秒)", 
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
                
                # 🎯 變數選擇
                numeric_columns = ptat_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_columns:
                    st.sidebar.markdown("#### 🎯 參數選擇")
                    
                    default_left_list = [c for c in numeric_columns if 'Temp' in c or 'temperature' in c.lower()]
                    default_left = default_left_list[0] if default_left_list else numeric_columns[0]
                    left_y_axis = st.sidebar.selectbox(
                        "📈 左側Y軸變數", 
                        options=numeric_columns, 
                        index=numeric_columns.index(default_left) if default_left in numeric_columns else 0
                    )
                    
                    right_y_axis_options = ['None'] + numeric_columns
                    default_right_list = [c for c in numeric_columns if 'Power' in c or 'power' in c.lower()]
                    default_right = default_right_list[0] if default_right_list else 'None'
                    try: 
                        default_right_index = right_y_axis_options.index(default_right)
                    except ValueError: 
                        default_right_index = 0
                    right_y_axis = st.sidebar.selectbox(
                        "📊 右側Y軸變數 (可選)", 
                        options=right_y_axis_options, 
                        index=default_right_index
                    )
                    
                    # 🎚️ Y軸範圍設定
                    st.sidebar.markdown("#### 🎚️ Y軸範圍")
                    auto_y = st.sidebar.checkbox("🔄 自動Y軸範圍", value=True)
                    y_limits = None
                    
                    if not auto_y and left_y_axis:
                        left_data = pd.to_numeric(ptat_df[left_y_axis], errors='coerce').dropna()
                        if len(left_data) > 0:
                            data_min, data_max = float(left_data.min()), float(left_data.max())
                            data_range = data_max - data_min
                            buffer = data_range * 0.1 if data_range > 0 else 1
                            y_min, y_max = st.sidebar.slider(
                                f"📊 {left_y_axis} 範圍",
                                data_min - buffer,
                                data_max + buffer,
                                (data_min - buffer, data_max + buffer),
                                step=0.1
                            )
                            y_limits = (y_min, y_max)
                    
                    # 🏠 主要內容區域
                    st.markdown("### 🔬 PTAT Log 數據分析")
                    
                    # 📈 圖表顯示
                    fig = generate_flexible_chart(ptat_df, left_y_axis, right_y_axis, x_limits, y_limits)
                    if fig: 
                        st.pyplot(fig, use_container_width=True)
                        
                        # 📊 PTAT Log 專用統計表格
                        st.markdown("### 📊 PTAT Log 統計分析")
                        
                        freq_df, power_df, temp_df = calculate_ptat_stats(ptat_df, x_limits)
                        
                        # 使用美化的分欄布局顯示三個表格
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("#### 🖥️ CPU Core Frequency")
                            if freq_df is not None and not freq_df.empty:
                                st.dataframe(freq_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("""
                                <div class="info-box">
                                    ❓ 未找到CPU頻率數據
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("#### ⚡ Package Power")
                            if power_df is not None and not power_df.empty:
                                st.dataframe(power_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("""
                                <div class="info-box">
                                    ❓ 未找到Package Power數據
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown("#### 🌡️ MSR Package Temp")
                            if temp_df is not None and not temp_df.empty:
                                st.dataframe(temp_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("""
                                <div class="info-box">
                                    ❓ 未找到MSR Package Temperature數據
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ 無法產生圖表")
                else:
                    st.warning("⚠️ 無可用的數值型數據")
            
            else:
                # 🔀 多檔案混合分析模式
                master_df = pd.concat(all_dfs)
                master_df_resampled = master_df.select_dtypes(include=['number']).resample('1S').mean(numeric_only=True).interpolate(method='linear')
                numeric_columns = master_df_resampled.columns.tolist()

                if numeric_columns:
                    st.sidebar.markdown("### ⚙️ 圖表設定")
                    
                    # 🎯 變數選擇
                    default_left_list = [c for c in numeric_columns if 'Temp' in c or 'T_' in c or 'CPU' in c]
                    default_left = default_left_list[0] if default_left_list else numeric_columns[0]
                    left_y_axis = st.sidebar.selectbox(
                        "📈 左側Y軸變數", 
                        options=numeric_columns, 
                        index=numeric_columns.index(default_left) if default_left in numeric_columns else 0
                    )
                    
                    right_y_axis_options = ['None'] + numeric_columns
                    default_right_index = 0
                    if len(numeric_columns) > 1:
                        default_right_list = [c for c in numeric_columns if 'Power' in c or 'Watt' in c or 'P_' in c]
                        default_right = default_right_list[0] if default_right_list else 'None'
                        try: 
                            default_right_index = right_y_axis_options.index(default_right)
                        except ValueError: 
                            default_right_index = 1
                    right_y_axis = st.sidebar.selectbox(
                        "📊 右側Y軸變數 (可選)", 
                        options=right_y_axis_options, 
                        index=default_right_index
                    )
                    
                    # 🎚️ X軸和Y軸範圍設定
                    st.sidebar.markdown("#### 🎚️ 軸範圍設定")
                    x_min_val = master_df_resampled.index.min().total_seconds()
                    x_max_val = master_df_resampled.index.max().total_seconds()
                    
                    if x_min_val < x_max_val:
                        x_min, x_max = st.sidebar.slider(
                            "⏱️ 時間範圍 (秒)", 
                            x_min_val, 
                            x_max_val, 
                            (x_min_val, x_max_val)
                        )
                    else:
                        x_min, x_max = x_min_val, x_max_val
                    
                    # Y軸範圍設定
                    auto_y = st.sidebar.checkbox("🔄 自動Y軸範圍", value=True)
                    y_limits = None
                    if not auto_y and left_y_axis:
                        left_data = pd.to_numeric(master_df_resampled[left_y_axis], errors='coerce').dropna()
                        if len(left_data) > 0:
                            data_min, data_max = float(left_data.min()), float(left_data.max())
                            data_range = data_max - data_min
                            buffer = data_range * 0.1 if data_range > 0 else 1
                            y_min, y_max = st.sidebar.slider(
                                f"📊 {left_y_axis} 範圍",
                                data_min - buffer,
                                data_max + buffer,
                                (data_min - buffer, data_max + buffer),
                                step=0.1
                            )
                            y_limits = (y_min, y_max)
                    
                    # 🏠 主要內容
                    st.markdown("### 🔀 動態比較圖表")
                    
                    fig = generate_flexible_chart(master_df_resampled, left_y_axis, right_y_axis, (x_min, x_max), y_limits)
                    if fig: 
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ 無法產生圖表")
                else:
                    st.warning("⚠️ 無可用的數值型數據進行繪圖")
        else:
            st.markdown("""
            <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 2rem; border-radius: 10px; text-align: center;">
                <h3>❌ 所有檔案解析失敗</h3>
                <p>請檢查檔案格式是否正確，或聯繫技術支援</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background-color: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h3>🚀 開始使用</h3>
        <p><strong>請在左側上傳您的 Log 文件開始分析</strong></p>
        
        <h4>📋 支援格式</h4>
        <ul>
            <li><strong>YOKOGAWA Excel (.xlsx)</strong> - 自動識別CH編號與Tag標籤</li>
            <li><strong>PTAT CSV (.csv)</strong> - CPU溫度、頻率、功耗分析</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)  
    
    # 📞 支援資訊
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📞 技術支援
    
    **需要幫助嗎？**
    - 📧 Email: support@example.com
    - 📱 Tel: +886-xxx-xxxx
    - 💬 即時聊天: 點擊右下角
    
    **📚 使用說明**
    - [📖 用戶手冊](https://example.com/manual)
    - [🎥 教學影片](https://example.com/videos)
    """)

# 🔚 頁面底部
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
    📊 通用數據分析平台 {VERSION} | 由 Streamlit 驅動 | © 2025 版權所有
</div>
""", unsafe_allow_html=True)
