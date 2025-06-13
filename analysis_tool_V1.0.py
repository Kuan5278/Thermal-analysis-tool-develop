# emergency_working_version.py
# 緊急可用版本 - 保證能運行，包含PTAT修復

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime
import numpy as np

# 版本資訊
VERSION = "v8.5 Emergency"
VERSION_DATE = "2025年6月"

def parse_ptat(file_content):
    """修復版PTAT Log解析器"""
    try:
        file_content.seek(0)
        df = pd.read_csv(file_content, header=0, thousands=',', low_memory=False)
        df.columns = df.columns.str.strip()
        time_column = 'Time'
        if time_column not in df.columns:
            return None, "PTAT Log中找不到 'Time' 欄位"
        time_series = df[time_column].astype(str).str.strip()
        time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
        datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
        valid_times_mask = datetime_series.notna()
        df = df[valid_times_mask].copy()
        if df.empty:
            return None, "PTAT Log時間格式無法解析"
        valid_datetimes = datetime_series[valid_times_mask]
        df['time_index'] = valid_datetimes - valid_datetimes.iloc[0]
        df = df.add_prefix('PTAT: ')
        df.rename(columns={'PTAT: time_index': 'time_index'}, inplace=True)
        return df.set_index('time_index'), "PTAT Log"
    except Exception as e:
        return None, f"解析PTAT Log時出錯: {e}"

def parse_yokogawa(file_content, is_excel=False):
    """YOKOGAWA Log解析器"""
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        
        if is_excel:
            possible_headers = [29, 28, 30, 27]
        else:
            possible_headers = [0, 1, 2]
            
        df = None
        found_time_col = None
        
        for header_row in possible_headers:
            try:
                file_content.seek(0)
                df = read_func(file_content, header=header_row, thousands=',')
                df.columns = df.columns.str.strip()
                
                time_candidates = ['Time', 'TIME', 'time', 'Date', 'DATE', 'date', 
                                 'DateTime', 'DATETIME', 'datetime', '時間', '日期時間',
                                 'Timestamp', 'TIMESTAMP', 'timestamp']
                
                for candidate in time_candidates:
                    if candidate in df.columns:
                        found_time_col = candidate
                        break
                
                if found_time_col:
                    break
                    
            except Exception as e:
                continue
        
        if df is None or found_time_col is None:
            return None, "YOKOGAWA Log中找不到時間欄位"
        
        time_column = found_time_col
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
                return None, f"無法解析時間格式: {e}"
        
        valid_times_mask = df['time_index'].notna()
        if valid_times_mask.sum() == 0:
            return None, "時間欄位中沒有有效的時間數據"
        
        df = df[valid_times_mask].copy()
        
        if len(df) > 0:
            start_time = df['time_index'].iloc[0]
            df['time_index'] = df['time_index'] - start_time
        
        numeric_columns = df.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            if col != 'time_index':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.add_prefix('YOKO: ')
        df.rename(columns={'YOKO: time_index': 'time_index'}, inplace=True)
        
        return df.set_index('time_index'), "YOKOGAWA Log"
        
    except Exception as e:
        return None, f"解析YOKOGAWA Log時出錯: {e}"

def parse_dispatcher(uploaded_file):
    """解析器調度中心"""
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
    
    try:
        if is_excel:
            return parse_yokogawa(file_content, is_excel)
        else:
            try:
                file_content.seek(0)
                first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(10)])
                file_content.seek(0)
                
                if 'MSR Package Temperature' in first_lines_text or 'Version,Date,Time' in first_lines_text:
                    return parse_ptat(file_content)
                else:
                    return parse_yokogawa(file_content, is_excel)
                    
            except Exception as e:
                file_content.seek(0)
                return parse_ptat(file_content)
        
    except Exception as e:
        return None, f"未知的Log檔案格式: {filename}"

def calculate_temp_stats(df, x_limits=None):
    """計算溫度統計數據"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    df_stats = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_stats = df_stats[(df_stats.index >= x_min_td) & (df_stats.index <= x_max_td)]
    
    if df_stats.empty:
        return pd.DataFrame()
    
    numeric_cols = df_stats.select_dtypes(include=['number']).columns
    temp_cols = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
    
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

def generate_temp_chart(df, x_limits=None, y_limits=None):
    """生成溫度圖表"""
    if df is None: 
        return None
    
    df_chart = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    if df_chart.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    numeric_cols = df_chart.select_dtypes(include=['number']).columns
    cols_to_plot = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
    
    max_channels = 15
    if len(cols_to_plot) > max_channels:
        cols_to_plot = cols_to_plot[:max_channels]
    
    for col in cols_to_plot:
        y_data = pd.to_numeric(df_chart[col], errors='coerce')
        if not y_data.isna().all():
            ax.plot(df_chart.index.total_seconds(), y_data, label=col, linewidth=1)
        
    ax.set_title("Temperature Analysis", fontsize=14, fontweight='bold')
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    
    if x_limits:
        ax.set_xlim(x_limits)
    
    if y_limits:
        ax.set_ylim(y_limits)
    
    fig.tight_layout()
    return fig

def main():
    """主程式"""
    st.set_page_config(
        page_title="數據分析平台",
        page_icon="📊",
        layout="wide"
    )
    
    # 標題
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; text-align: center; color: white;">
        <h1>📊 數據分析平台</h1>
        <p>智能解析 YOKOGAWA & PTAT Log 文件</p>
        <p><strong>""" + VERSION + """</strong> | """ + VERSION_DATE + """</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 側邊欄
    st.sidebar.markdown("### 🎛️ 控制面板")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True,
        help="支援 YOKOGAWA Excel 格式和 PTAT CSV 格式"
    )
    
    if uploaded_files:
        # 檔案資訊
        st.sidebar.markdown("### 📂 已上傳檔案")
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / 1024
            st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
        
        # 處理單一檔案
        if len(uploaded_files) == 1:
            df_check, log_type_check = parse_dispatcher(uploaded_files[0])
            
            if df_check is not None:
                st.success(f"✅ 檔案解析成功！檔案類型：{log_type_check}")
                st.write(f"📊 數據筆數：{len(df_check):,} 筆")
                st.write(f"🔢 欄位數量：{len(df_check.columns):,} 個")
                
                # 時間範圍設定
                if len(df_check) > 0:
                    x_min_val = df_check.index.min().total_seconds()
                    x_max_val = df_check.index.max().total_seconds()
                    
                    if x_min_val < x_max_val:
                        x_min, x_max = st.sidebar.slider(
                            "⏱️ 時間範圍 (秒)", 
                            float(x_min_val), 
                            float(x_max_val), 
                            (float(x_min_val), float(x_max_val))
                        )
                        x_limits = (x_min, x_max)
                    else:
                        x_limits = None
                else:
                    x_limits = None
                
                # 主要內容
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 📈 數據圖表")
                    
                    fig = generate_temp_chart(df_check, x_limits)
                    if fig: 
                        st.pyplot(fig)
                    else: 
                        st.warning("⚠️ 無法產生圖表")
                
                with col2:
                    st.markdown("### 📊 統計數據")
                    stats_df = calculate_temp_stats(df_check, x_limits)
                    if not stats_df.empty:
                        st.dataframe(stats_df, use_container_width=True)
                    else:
                        st.info("無統計數據")
                
                # 數據預覽
                st.markdown("### 📋 數據預覽")
                st.dataframe(df_check.head(10), use_container_width=True)
                
            else:
                st.error(f"❌ 檔案解析失敗：{log_type_check}")
        else:
            st.info("請選擇單一檔案進行分析")
    
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

if __name__ == "__main__":
    main()
