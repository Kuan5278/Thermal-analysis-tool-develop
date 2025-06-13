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
        
        # 修正為TimedeltaIndex
        timedelta_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
        df.dropna(subset=[timedelta_series], inplace=True)
        df['time_index'] = timedelta_series - timedelta_series.iloc[0]

        df = df.add_prefix('PTAT: ')
        df.rename(columns={'PTAT: time_index': 'time_index'}, inplace=True)
        return df.set_index('time_index'), "PTAT Log"
    except Exception as e:
        return None, f"解析PTAT Log時出錯: {e}"

# --- 子模組：YOKOGAWA Log 解析器 ---
def parse_yokogawa(file_content, is_excel=False):
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        
        ch_list = read_func(file_content, header=None, skiprows=27, nrows=1).iloc[0].astype(str).tolist()
        file_content.seek(0)
        tag_list = read_func(file_content, header=None, skiprows=28, nrows=1).iloc[0].astype(str).tolist()
        file_content.seek(0)

        final_columns = [tag if pd.notna(tag) and str(tag).strip() not in ['', 'nan'] else ch for tag, ch in zip(tag_list, ch_list)]
        
        df = read_func(file_content, header=None, skiprows=29)
        df.columns = final_columns
        df.columns = df.columns.str.strip()
        
        # 修正：時間欄位從 'TIME' 改為 'RT'
        time_column = 'RT'
        if time_column not in df.columns: return None, f"YOKOGAWA Log中找不到 '{time_column}' 欄位"
        
        # 將相對時間(秒)轉換為TimedeltaIndex
        df['time_index'] = pd.to_timedelta(pd.to_numeric(df[time_column], errors='coerce'), unit='s')
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
    
    # 嗅探檔案內容
    if is_excel:
        # 對於Excel，需要先讀取才能嗅探
        try:
            df_sniff = pd.read_excel(file_content, header=None, skiprows=27, nrows=1)
            if 'CH001' in df_sniff.iloc[0].astype(str).values:
                file_content.seek(0)
                return parse_yokogawa(file_content, is_excel=True)
        except Exception:
             pass # 如果讀取失敗，就當作不是YOKOGAWA
    else: # CSV
        first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(30)])
        file_content.seek(0)
        if 'MSR Package Temperature' in first_lines_text:
            return parse_ptat(file_content)
        elif 'CH001' in first_lines_text:
            return parse_yokogawa(file_content, is_excel=False)
            
    return None, "未知的Log檔案格式"

# --- 圖表繪製函式 ---
def generate_flexible_chart(df, left_col, right_col, x_limits):
    if df is None or left_col is None: return None
    df_chart = df.copy()
    
    x_min_td = pd.to_timedelta(x_limits[0], unit='s')
    x_max_td = pd.to_timedelta(x_limits[1], unit='s')
    
    if x_limits: df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col and right_col != 'None':
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=16)
    
    x_axis_seconds = df_chart.index.total_seconds()
    color = 'tab:blue'
    ax1.set_xlabel('Elapsed Time (seconds)', fontsize=12)
    ax1.set_ylabel(left_col, color=color, fontsize=12)
    ax1.plot(x_axis_seconds, df_chart['left_val'], color=color)
    ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', linewidth=0.5)

    if right_col and right_col != 'None':
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(right_col, color=color, fontsize=12)
        ax2.plot(x_axis_seconds, df_chart['right_val'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    if x_limits: ax1.set_xlim(x_limits)
    fig.tight_layout()
    return fig

# --- Streamlit 網頁應用程式介面 ---
st.set_page_config(layout="wide")
st.title("通用數據分析平台")

st.sidebar.header("控制面板")
uploaded_files = st.sidebar.file_uploader("上傳Log File (可多選)", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    all_dfs = []
    log_types_detected = []
    
    with st.spinner('正在解析所有Log檔案...'):
        for file in uploaded_files:
            df, log_type = parse_dispatcher(file)
            if df is not None:
                all_dfs.append(df)
                log_types_detected.append(f"{file.name} (辨識為: {log_type})")
            else:
                st.error(f"檔案 '{file.name}' 解析失敗: {log_type}")

    if all_dfs:
        st.sidebar.success("檔案解析完成！")
        for name in log_types_detected:
            st.sidebar.markdown(f"- `{name}`")

        # 修正：在resample().mean()中加入 numeric_only=True
        master_df = pd.concat(all_dfs).sort_index()
        master_df_resampled = master_df.resample('1S').mean(numeric_only=True).interpolate(method='linear')
        
        numeric_columns = master_df_resampled.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_columns:
            st.sidebar.header("圖表設定")
            left_y_axis = st.sidebar.selectbox("選擇左側Y軸變數", options=numeric_columns, index=0)
            right_y_axis_options = ['None'] + numeric_columns
            right_y_axis = st.sidebar.selectbox("選擇右側Y軸變數 (可不選)", options=right_y_axis_options, index=1 if len(numeric_columns) > 1 else 0)

            st.sidebar.header("X軸範圍設定 (秒)")
            x_min_val = master_df_resampled.index.min().total_seconds()
            x_max_val = master_df_resampled.index.max().total_seconds()
            x_min = st.sidebar.number_input("最小值", value=x_min_val)
            x_max = st.sidebar.number_input("最大值", value=x_max_val)
            
            st.header("動態比較圖表")
            fig = generate_flexible_chart(master_df_resampled, left_y_axis, right_y_axis, (x_min, x_max))
            
            if fig: st.pyplot(fig)
        else:
            st.warning("所有檔案解析後，無可用的數值型數據進行繪圖。")
else:
    st.sidebar.info("請上傳您的 Log File(s) 開始分析。")
