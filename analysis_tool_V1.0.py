# 修復語法錯誤的通用分析平台
# 這個版本專門修復三引號字符串的語法問題

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime
import numpy as np

# 檢查語法錯誤的常見位置
def check_syntax_issues():
    """檢查代碼中的語法問題"""
    syntax_tips = [
        "1. 檢查所有三引號字符串是否正確關閉",
        "2. 確保 st.markdown 的參數正確",
        "3. 檢查縮進是否一致",
        "4. 確保所有括號都有對應的關閉",
        "5. 檢查字符串中是否有未轉義的引號"
    ]
    return syntax_tips

# --- 版本資訊設定 ---
VERSION = "v8.0 Fixed"
VERSION_DATE = "2025年6月"

# 🔧 修復常見的語法錯誤模式
def display_safe_markdown(content):
    """安全的markdown顯示函數"""
    try:
        st.markdown(content, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Markdown顯示錯誤: {str(e)}")
        st.text(content)  # 備用顯示方式

# --- 原有的解析函數 (保持不變) ---
def parse_ptat(file_content):
    """
    修復版PTAT Log解析器 v8.5
    專門處理格式: Version,Date,Time,Relative Time(mS),Diff time(mS),CPUID,Stepping,Graphics ID...
    """
    try:
        # 重置文件指針
        file_content.seek(0)
        
        # 使用更寬容的參數讀取CSV
        df = pd.read_csv(
            file_content, 
            header=0,  # 第一行是標題
            thousands=',', 
            low_memory=False,
            encoding='utf-8',  # 明確指定編碼
            sep=',',  # 明確指定分隔符
            on_bad_lines='skip',  # 跳過格式錯誤的行
            dtype=str  # 先全部讀取為字符串，後續再轉換
        )
        
        # 清理欄位名稱
        df.columns = df.columns.str.strip()
        
        # 檢查是否有Time欄位
        time_column = 'Time'
        if time_column not in df.columns: 
            return None, "PTAT Log中找不到 'Time' 欄位"
        
        # 處理時間數據
        time_series = df[time_column].astype(str).str.strip()
        
        # 處理時間格式 - 支援多種格式
        try:
            # 方法1: 嘗試標準時間格式 HH:MM:SS.fff
            time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
            datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
            
            # 方法2: 如果上面失敗，嘗試 HH:MM:SS
            if datetime_series.isna().all():
                datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S', errors='coerce')
            
            # 方法3: 如果還是失敗，嘗試自動推斷
            if datetime_series.isna().all():
                datetime_series = pd.to_datetime(time_series, errors='coerce', infer_datetime_format=True)
            
            # 方法4: 最後嘗試，如果是數字格式（秒數）
            if datetime_series.isna().all():
                try:
                    numeric_time = pd.to_numeric(time_series, errors='coerce')
                    if not numeric_time.isna().all():
                        # 假設是秒數，轉換為timedelta
                        datetime_series = pd.to_timedelta(numeric_time, unit='s')
                except:
                    pass
                    
        except Exception as e:
            return None, f"時間格式解析失敗: {str(e)}"
        
        # 檢查時間解析結果
        valid_times_mask = datetime_series.notna()
        if valid_times_mask.sum() == 0:
            return None, "PTAT Log時間格式無法解析 - 所有時間數據都無效"
        
        # 過濾有效數據
        df = df[valid_times_mask].copy()
        if df.empty: 
            return None, "過濾後沒有有效的時間數據"
        
        # 計算相對時間
        valid_datetimes = datetime_series[valid_times_mask]
        
        # 如果是timestamp格式，轉換為相對時間
        if hasattr(valid_datetimes.iloc[0], 'to_pydatetime'):
            df['time_index'] = valid_datetimes - valid_datetimes.iloc[0]
        else:
            # 如果已經是timedelta格式
            df['time_index'] = valid_datetimes
        
        # 數據清理：轉換數值欄位
        for col in df.columns:
            if col not in ['time_index', 'Version', 'Date', 'Time', 'CPU Name', 'CPU Brand String', 'Host ID']:
                try:
                    # 嘗試轉換為數值
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    # 如果轉換成功且不是全部NaN，則使用數值版本
                    if not numeric_data.isna().all():
                        df[col] = numeric_data
                except:
                    # 轉換失敗就保持原樣
                    pass
        
        # 添加PTAT前綴
        df = df.add_prefix('PTAT: ')
        df.rename(columns={'PTAT: time_index': 'time_index'}, inplace=True)
        
        # 設置時間索引
        result_df = df.set_index('time_index')
        
        return result_df, "PTAT Log"
        
    except pd.errors.EmptyDataError:
        return None, "PTAT Log檔案為空或格式錯誤"
    except pd.errors.ParserError as e:
        return None, f"PTAT Log解析錯誤: {str(e)}"
    except UnicodeDecodeError:
        # 如果UTF-8失敗，嘗試其他編碼
        try:
            file_content.seek(0)
            df = pd.read_csv(
                file_content, 
                header=0,
                thousands=',', 
                low_memory=False,
                encoding='latin1',  # 嘗試latin1編碼
                sep=',',
                on_bad_lines='skip',
                dtype=str
            )
            # 重複處理邏輯
            df.columns = df.columns.str.strip()
            time_column = 'Time'
            if time_column not in df.columns: 
                return None, "PTAT Log中找不到 'Time' 欄位"
            
            time_series = df[time_column].astype(str).str.strip()
            time_series_cleaned = time_series.str.replace(r':(\d{3})$', r'.\1', regex=True)
            datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S.%f', errors='coerce')
            
            if datetime_series.isna().all():
                datetime_series = pd.to_datetime(time_series_cleaned, format='%H:%M:%S', errors='coerce')
            
            valid_times_mask = datetime_series.notna()
            if valid_times_mask.sum() == 0:
                return None, "PTAT Log時間格式無法解析"
            
            df = df[valid_times_mask].copy()
            valid_datetimes = datetime_series[valid_times_mask]
            df['time_index'] = valid_datetimes - valid_datetimes.iloc[0]
            
            for col in df.columns:
                if col not in ['time_index', 'Version', 'Date', 'Time', 'CPU Name', 'CPU Brand String', 'Host ID']:
                    try:
                        numeric_data = pd.to_numeric(df[col], errors='coerce')
                        if not numeric_data.isna().all():
                            df[col] = numeric_data
                    except:
                        pass
            
            df = df.add_prefix('PTAT: ')
            df.rename(columns={'PTAT: time_index': 'time_index'}, inplace=True)
            result_df = df.set_index('time_index')
            
            return result_df, "PTAT Log"
            
        except Exception as e:
            return None, f"編碼錯誤無法修復: {str(e)}"
    except Exception as e:
        return None, f"解析PTAT Log時出現未知錯誤: {str(e)}"
                else:
                    pass
                    
            except Exception as e:
                pass
        
    except Exception as e:
        pass
        
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

def generate_yokogawa_temp_chart(df, x_limits=None, y_limits=None):
    """生成YOKOGAWA溫度圖表"""
    if df is None: 
        return None
    
    df_chart = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    if df_chart.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    
    numeric_cols = df_chart.select_dtypes(include=['number']).columns
    cols_to_plot = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
    
    max_channels = 15
    if len(cols_to_plot) > max_channels:
        cols_to_plot = cols_to_plot[:max_channels]
    
    for col in cols_to_plot:
        y_data = pd.to_numeric(df_chart[col], errors='coerce')
        if not y_data.isna().all():
            ax.plot(df_chart.index.total_seconds(), y_data, label=col, linewidth=1)
        
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

# --- 主應用程式 ---
def main():
    st.set_page_config(
        page_title="通用數據分析平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 🔧 修復: 使用安全的CSS樣式
    css_styles = """
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
    </style>
    """
    
    # 🔧 修復: 安全地顯示CSS
    display_safe_markdown(css_styles)
    
    # 🔧 修復: 使用安全的主標題
    main_header = """
    <div class="main-header">
        <h1>📊 通用數據分析平台</h1>
        <p>智能解析 YOKOGAWA & PTAT Log 文件，提供專業級數據分析與視覺化</p>
    </div>
    """
    display_safe_markdown(main_header)
    
    # 版本資訊
    with st.expander("📋 版本資訊", expanded=False):
        st.write(f"**當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**")
        st.write("### 🆕 修復內容：")
        st.write("- 🔧 修復三引號字符串語法錯誤")
        st.write("- ✅ 確保所有markdown字符串正確關閉")
        st.write("- 🛡️ 增加安全的markdown顯示函數")
        st.write("- 📝 改善錯誤處理機制")
    
    # 側邊欄
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True,
        help="支援 YOKOGAWA Excel 格式和 PTAT CSV 格式"
    )
    
    if uploaded_files:
        # 檔案資訊顯示
        st.sidebar.markdown("### 📂 已上傳檔案")
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / 1024
            st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
        
        st.sidebar.markdown("---")
        
        # 檔案解析
        if len(uploaded_files) == 1:
            df_check, log_type_check = parse_dispatcher(uploaded_files[0])
            is_single_yokogawa = (log_type_check == "YOKOGAWA Log")
        else:
            is_single_yokogawa = False
        
        # YOKOGAWA 專屬顯示模式
        if is_single_yokogawa:
            # 🔧 修復: 使用安全的成功信息顯示
            success_message = f"""
            <div class="success-box">
                <strong>✅ 檔案解析成功</strong><br>
                📄 檔案類型：{log_type_check}<br>
                📊 數據筆數：{len(df_check):,} 筆<br>
                🔢 通道數量：{len([c for c in df_check.columns if df_check[c].dtype in ['float64', 'int64']]):,} 個
            </div>
            """
            display_safe_markdown(success_message)
            
            # 圖表設定
            st.sidebar.markdown("### ⚙️ 圖表設定")
            
            if df_check is not None and len(df_check) > 0:
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
                
                # Y軸範圍設定
                st.sidebar.markdown("#### 🎯 Y軸溫度範圍")
                df_temp = df_check.copy()
                if x_limits:
                    x_min_td = pd.to_timedelta(x_limits[0], unit='s')
                    x_max_td = pd.to_timedelta(x_limits[1], unit='s')
                    df_temp = df_temp[(df_temp.index >= x_min_td) & (df_temp.index <= x_max_td)]
                
                if not df_temp.empty:
                    numeric_cols = df_temp.select_dtypes(include=['number']).columns
                    temp_cols = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
                    
                    if temp_cols:
                        all_temps = pd.concat([pd.to_numeric(df_temp[col], errors='coerce') for col in temp_cols])
                        all_temps = all_temps.dropna()
                        
                        if len(all_temps) > 0:
                            temp_min = float(all_temps.min())
                            temp_max = float(all_temps.max())
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
                            
                            # 🔧 修復: 使用安全的範圍信息顯示
                            range_info = f"""
                            **📈 當前溫度範圍：**
                            - 最高：{temp_max:.1f}°C
                            - 最低：{temp_min:.1f}°C
                            - 差值：{temp_range:.1f}°C
                            """
                            st.sidebar.markdown(range_info)
                        else:
                            y_limits = None
                    else:
                        y_limits = None
                else:
                    y_limits = None
            else:
                x_limits = None
                y_limits = None
            
            # 主要內容區域
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 📈 YOKOGAWA 全通道溫度曲線圖")
                
                if df_check is not None:
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
                    
                    if len(stats_df) > 0:
                        try:
                            max_temps = [float(x.replace('°C', '')) for x in stats_df['Tmax (°C)'] if x != 'N/A']
                            avg_temps = [float(x.replace('°C', '')) for x in stats_df['Tavg (°C)'] if x != 'N/A']
                            
                            if max_temps and avg_temps:
                                # 🔧 修復: 使用安全的統計摘要顯示
                                summary_info = f"""
                                <div class="metric-card">
                                    <strong>🔥 整體最高溫：</strong> {max(max_temps):.1f}°C<br>
                                    <strong>📊 平均溫度：</strong> {sum(avg_temps)/len(avg_temps):.1f}°C<br>
                                    <strong>📈 活躍通道：</strong> {len(stats_df)} 個
                                </div>
                                """
                                display_safe_markdown(summary_info)
                        except:
                            pass
                else:
                    # 🔧 修復: 使用安全的無數據信息顯示
                    no_data_info = """
                    <div class="info-box">
                        ❓ 無統計數據可顯示<br>
                        請檢查時間範圍設定
                    </div>
                    """
                    display_safe_markdown(no_data_info)
        
        else:
            st.info("多檔案分析功能開發中...")
    
    else:
        # 🔧 修復: 使用安全的歡迎頁面
        welcome_info = """
        <div class="info-box">
            <h3>🚀 開始使用</h3>
            <p><strong>請在左側上傳您的 Log 文件開始分析</strong></p>
            
            <h4>📋 支援格式</h4>
            <ul>
                <li><strong>YOKOGAWA Excel (.xlsx)</strong> - 自動識別CH編號與Tag標籤</li>
                <li><strong>PTAT CSV (.csv)</strong> - CPU溫度、頻率、功耗分析</li>
            </ul>
            
            <h4>✨ 主要功能</h4>
            <ul>
                <li>🎯 智能檔案格式識別</li>
                <li>📊 即時數據統計分析</li>
                <li>📈 動態圖表與範圍調整</li>
                <li>🔄 多檔案混合比較</li>
            </ul>
        </div>
        """
        display_safe_markdown(welcome_info)
        
        # 語法檢查提示
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 語法檢查")
        syntax_tips = check_syntax_issues()
        for tip in syntax_tips:
            st.sidebar.text(tip)
    
    # 🔧 修復: 使用安全的頁面底部
    footer_info = f"""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
        📊 通用數據分析平台 {VERSION} | 語法修復版 | © 2025 版權所有
    </div>
    """
    display_safe_markdown(footer_info)

# 執行主程式
if __name__ == "__main__":
    main()
