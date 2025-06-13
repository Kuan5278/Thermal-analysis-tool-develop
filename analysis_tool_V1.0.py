# universal_analysis_platform.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- 子模組：PTAT Log 解析器 ---
def parse_ptat(file_content):
    """專門解析 PTAT Log File"""
    try:
        df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
        df.columns = df.columns.str.strip()
        time_column = 'Time'
        if time_column not in df.columns: return None, f"PTAT Log中找不到 '{time_column}' 欄位"

        time_series = df[time_column].astype(str).str.strip()
        time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
        df['Time_Cleaned'] = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
        df.dropna(subset=['Time_Cleaned'], inplace=True)
        if df.empty: return None, "PTAT Log時間格式無法解析"

        start_time = df['Time_Cleaned'].iloc[0]
        df['Elapsed Time (s)'] = (df['Time_Cleaned'] - start_time).dt.total_seconds()
        
        df = df.add_prefix('PTAT: ')
        df.rename(columns={'PTAT: Elapsed Time (s)': 'Elapsed Time (s)'}, inplace=True)
        return df.set_index('Elapsed Time (s)'), "PTAT Log"
    except Exception as e:
        return None, f"解析PTAT Log時出錯: {e}"

# --- 子模組：YOKOGAWA Log 解析器 ---
def parse_yokogawa(file_content):
    """專門解析 YOKOGAWA Log File"""
    try:
        # 讀取 Channel ID (第28行) 和 Tag (第29行)
        df_ch = pd.read_csv(file_content, header=None, skiprows=27, nrows=1, low_memory=False)
        file_content.seek(0) # 重設指標
        df_tag = pd.read_csv(file_content, header=None, skiprows=28, nrows=1, low_memory=False)
        file_content.seek(0) # 重設指標

        ch_list = df_ch.iloc[0].astype(str).tolist()
        tag_list = df_tag.iloc[0].astype(str).tolist()

        # 備用邏輯：如果Tag是空的，就用Channel ID替代
        final_columns = [tag if pd.notna(tag) and str(tag).strip() != '' else ch for tag, ch in zip(tag_list, ch_list)]
        
        # 讀取真正的數據 (從第30行開始)
        df = pd.read_csv(file_content, header=None, skiprows=29, low_memory=False)
        df.columns = final_columns
        df.columns = df.columns.str.strip()

        time_column = 'TIME'
        if time_column not in df.columns: return None, f"YOKOGAWA Log中找不到 '{time_column}' 欄位"
        
        # 清理與轉換時間
        time_series = df[time_column].astype(str).str.strip()
        df['Time_Cleaned'] = pd.to_datetime(time_series, errors='coerce')
        df.dropna(subset=['Time_Cleaned'], inplace=True)
        if df.empty: return None, "YOKOGAWA Log時間格式無法解析"

        start_time = df['Time_Cleaned'].iloc[0]
        df['Elapsed Time (s)'] = (df['Time_Cleaned'] - start_time).dt.total_seconds()

        df = df.add_prefix('YOKO: ')
        df.rename(columns={'YOKO: Elapsed Time (s)': 'Elapsed Time (s)'}, inplace=True)
        return df.set_index('Elapsed Time (s)'), "YOKOGAWA Log"
    except Exception as e:
        return None, f"解析YOKOGAWA Log時出錯: {e}"

# --- 主模組：解析器調度中心 ---
def parse_dispatcher(uploaded_file):
    """自動嗅探檔案類型並呼叫對應的解析器"""
    file_content = io.BytesIO(uploaded_file.getvalue())
    first_lines = [file_content.readline().decode('utf-8', errors='ignore').strip() for _ in range(30)]
    file_content.seek(0)
    
    full_text = "".join(first_lines)

    if 'MSR Package Temperature(Degree C)' in full_text and 'CPU0-Frequency(MHz)' in full_text:
        return parse_ptat(file_content)
    elif 'CH001' in full_text:
        return parse_yokogawa(file_content)
    else:
        return None, "未知的Log檔案格式"

# --- 動態圖表繪製函式 ---
def generate_flexible_chart(df, left_col, right_col, x_limits):
    if df is None or left_col is None: return None
    df_chart = df.copy()
    if x_limits: df_chart = df_chart.loc[x_limits[0]:x_limits[1]]
    
    # 轉換Y軸數值
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col and right_col != 'None':
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=16)
    
    color = 'tab:blue'
    ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12)
    ax1.set_ylabel(left_col, color=color, fontsize=12)
    ax1.plot(df_chart.index, df_chart['left_val'], color=color)
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', linewidth=0.5)

    if right_col and right_col != 'None':
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=12)
        ax2.plot(df_chart.index, df_chart['right_val'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    if x_limits: ax1.set_xlim(x_limits)
    fig.tight_layout()
    return fig

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide")
st.title("通用數據分析平台")

st.sidebar.header("控制面板")
uploaded_files = st.sidebar.file_uploader("上傳Log File (可多選)", type="csv", accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    log_types = []
    
    with st.spinner('正在解析所有Log檔案...'):
        for file in uploaded_files:
            df, log_type = parse_dispatcher(file)
            if df is not None:
                all_dfs.append(df)
                log_types.append(f"{file.name} (辨識為: {log_type})")
            else:
                st.error(f"檔案 '{file.name}' 解析失敗: {log_type}")

    if all_dfs:
        st.sidebar.success("檔案解析完成！")
        st.sidebar.write("已載入日誌:")
        for name in log_types:
            st.sidebar.markdown(f"- `{name}`")

        # --- 數據整合 ---
        # 將所有數據基於時間抽樣並合併
        master_df = pd.concat([df.resample('1S').mean() for df in all_dfs], axis=1)
        master_df.interpolate(method='linear', inplace=True) # 填補抽樣後的空值

        numeric_columns = master_df.select_dtypes(include=['number']).columns.tolist()
        
        st.sidebar.header("圖表設定")
        left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, index=0)
        right_y_axis_options = ['None'] + numeric_columns
        right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=1 if len(numeric_columns) > 1 else 0)

        st.sidebar.header("X軸範圍設定 (秒)")
        x_min = st.sidebar.number_input("最小值", value=master_df.index.min())
        x_max = st.sidebar.number_input("最大值", value=master_df.index.max())
        
        st.header("動態比較圖表")
        fig = generate_flexible_chart(master_df, left_y_axis, right_y_axis, (x_min, x_max))
        
        if fig:
            st.pyplot(fig)
        else:
            st.warning("無法產生圖表，請確認選擇的欄位。")
else:
    st.sidebar.info("請上傳您的 Log File 開始分析。")