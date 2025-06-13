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
    
    fig, ax1 = plt.subplots(figsize=(10.2, 5.1))
    plt.title(f'{left_col} {"& " + right_col if right_col and right_col != "None" else ""}', fontsize=14, fontweight='bold')
    
    x_axis_seconds = df_chart.index.total_seconds()
    color = 'tab:blue'
    ax1.set_xlabel('Elapsed Time (seconds)', fontsize=11)
    ax1.set_ylabel(left_col, color=color, fontsize=11)
    ax1.plot(x_axis_seconds, df_chart['left_val'], color=color, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
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
        st.markdown("💡 **使用提示：** 支援YOKOGAWA Excel格式、PTAT CSV格式，現在加入AI功能！")

# --- 主應用程式 ---
def main():
    st.set_page_config(
        page_title="AI-Ready 數據分析平台",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化AI組件
    ai_engine = BasicAIEngine()
    conversation_system = SimpleConversationSystem()
    
    # Session State 初始化
    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # 自定義CSS樣式
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .ai-feature-box {
            background: linear-gradient(145deg, #e3f2fd, #bbdefb);
            border: 2px solid #2196f3;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
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
    """, unsafe_allow_html=True)
    
    # 主頁面標題
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 AI-Ready 數據分析平台</h1>
        <p>智能解析 YOKOGAWA & PTAT Log 文件，現在加入AI功能！</p>
        <p><strong>{VERSION}</strong> | {VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 版本資訊區域
    display_version_info()
    
    # AI功能狀態顯示
    with st.expander("🤖 AI功能狀態", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if AI_ANOMALY_AVAILABLE:
                st.success("✅ ML異常檢測 已啟用")
                st.info("🔬 使用 Isolation Forest 算法")
            else:
                st.warning("⚠️ ML異常檢測 未啟用")
                st.info("💡 安装 scikit-learn 啟用高級功能")
        
        with col2:
            if PLOTLY_AVAILABLE:
                st.success("✅ 交互式圖表 已啟用")
                st.info("📊 支援 Plotly 圖表")
            else:
                st.warning("⚠️ 交互式圖表 未啟用")
                st.info("💡 安装 plotly 啟用交互功能")
        
        with col3:
            if SCIPY_AVAILABLE:
                st.success("✅ 高級統計 已啟用")
                st.info("📈 支援統計檢驗")
            else:
                st.warning("⚠️ 高級統計 未啟用")
                st.info("💡 安装 scipy 啟用統計功能")
    
    # 側邊欄設計
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True,
        help="支援 YOKOGAWA Excel 格式和 PTAT CSV 格式"
    )
    
    # AI功能選擇
    st.sidebar.markdown("### 🤖 AI功能選擇")
    ai_features = st.sidebar.multiselect(
        "選擇AI分析功能",
        ["🔍 基礎異常檢測", "🤖 ML異常檢測", "📈 趨勢分析", "💡 智能洞察"],
        default=["🔍 基礎異常檢測", "📈 趨勢分析"]
    )
    
    # 智能對話區域
    st.sidebar.markdown("### 💬 AI助手")
    user_query = st.sidebar.text_input(
        "問AI助手",
        placeholder="例如：檢測異常、分析趨勢...",
        key="ai_query"
    )
    
    if user_query:
        ai_response = conversation_system.process_query(user_query)
        st.sidebar.markdown(f"""
        <div class="ai-feature-box">
            <strong>🤖 AI回應：</strong><br>
            {ai_response}
        </div>
        """, unsafe_allow_html=True)
    
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
        
        # 創建頁面選項卡
        if is_single_yokogawa:
            tab1, tab2, tab3 = st.tabs(["📊 數據分析", "🤖 AI分析", "📈 交互圖表"])
        else:
            tab1, tab2, tab3 = st.tabs(["📊 數據分析", "🤖 AI分析", "📈 交互圖表"])
        
        with tab1:
            # 原有的數據分析功能
            if is_single_yokogawa:
                st.markdown(f"""
                <div class="success-box">
                    <strong>✅ 檔案解析成功</strong><br>
                    📄 檔案類型：{log_type_check}<br>
                    📊 數據筆數：{len(df_check):,} 筆<br>
                    🔢 通道數量：{len([c for c in df_check.columns if df_check[c].dtype in ['float64', 'int64']]):,} 個
                </div>
                """, unsafe_allow_html=True)
                
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
            
            else:
                # 處理PTAT或多檔案
                all_dfs = []
                log_types = []
                
                for file in uploaded_files:
                    df, log_type = parse_dispatcher(file)
                    if df is not None:
                        all_dfs.append(df)
                        log_types.append(log_type)
                
                if all_dfs:
                    # 檔案解析狀態
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
                        # 單一PTAT Log的特殊處理
                        ptat_df = all_dfs[0]
                        
                        st.sidebar.markdown("### ⚙️ PTAT 圖表設定")
                        
                        # 時間範圍設定
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
                        
                        # 變數選擇
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
                            
                            # Y軸範圍設定
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
                            
                            # 主要內容區域
                            st.markdown("### 🔬 PTAT Log 數據分析")
                            
                            # 圖表顯示
                            fig = generate_flexible_chart(ptat_df, left_y_axis, right_y_axis, x_limits, y_limits)
                            if fig: 
                                st.pyplot(fig, use_container_width=True)
                                
                                # PTAT Log 專用統計表格
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
        
        with tab2:
            st.markdown("### 🤖 AI智能分析")
            
            # 選擇分析的數據集
            all_dfs = []
            log_types = []
            
            for file in uploaded_files:
                df, log_type = parse_dispatcher(file)
                if df is not None:
                    all_dfs.append(df)
                    log_types.append(log_type)
            
            if all_dfs:
                if len(all_dfs) > 1:
                    selected_df_idx = st.selectbox(
                        "選擇數據集進行AI分析",
                        range(len(all_dfs)),
                        format_func=lambda x: f"{uploaded_files[x].name} ({log_types[x]})"
                    )
                    current_df = all_dfs[selected_df_idx]
                else:
                    current_df = all_dfs[0]
                    selected_df_idx = 0
                
                # 數據概況
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("數據筆數", f"{len(current_df):,}")
                with col2:
                    numeric_cols = current_df.select_dtypes(include=[np.number]).columns
                    st.metric("數值欄位", len(numeric_cols))
                with col3:
                    st.metric("AI功能", f"{len(ai_features)} 個已選")
                
                # AI分析執行區域
                st.markdown("#### 🤖 執行AI分析")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if "🔍 基礎異常檢測" in ai_features:
                        st.markdown("**🔍 基礎異常檢測**")
                        
                        # 選擇要檢測的欄位
                        detection_columns = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
                        if detection_columns:
                            selected_col = st.selectbox(
                                "選擇檢測欄位",
                                detection_columns,
                                key="basic_anomaly_col"
                            )
                            
                            threshold = st.slider(
                                "異常閾值倍數",
                                1.0, 5.0, 2.5, 0.1,
                                help="數值越大，檢測越嚴格",
                                key="basic_threshold"
                            )
                            
                            if st.button("🔍 執行基礎異常檢測", key="basic_anomaly"):
                                with st.spinner("正在分析..."):
                                    result, msg = ai_engine.basic_anomaly_detection(
                                        current_df[selected_col], 
                                        threshold
                                    )
                                    
                                    if result:
                                        st.success(msg)
                                        st.json(result)
                                        st.session_state.ai_analysis_results.append(f"基礎異常檢測: {msg}")
                                    else:
                                        st.error(msg)
                
                with col2:
                    if "🤖 ML異常檢測" in ai_features:
                        st.markdown("**🤖 ML異常檢測**")
                        
                        if AI_ANOMALY_AVAILABLE:
                            if st.button("🤖 執行ML異常檢測", key="ml_anomaly"):
                                with st.spinner("ML算法分析中..."):
                                    result_df, msg = ai_engine.advanced_anomaly_detection(current_df)
                                    
                                    if result_df is not None:
                                        st.success(msg)
                                        
                                        # 顯示異常點統計
                                        anomaly_points = result_df[result_df['is_anomaly'] == True]
                                        if len(anomaly_points) > 0:
                                            st.markdown(f"**發現 {len(anomaly_points)} 個異常點**")
                                            
                                            # 顯示異常分數分布
                                            fig, ax = plt.subplots(figsize=(8, 4))
                                            ax.hist(result_df['anomaly_score'].dropna(), bins=20, alpha=0.7)
                                            ax.set_title("異常分數分布")
                                            ax.set_xlabel("異常分數")
                                            ax.set_ylabel("頻次")
                                            st.pyplot(fig)
                                        else:
                                            st.info("🎉 未發現明顯異常")
                                        
                                        st.session_state.ai_analysis_results.append(f"ML異常檢測: {msg}")
                                    else:
                                        st.error(msg)
                        else:
                            st.warning("需要安裝 scikit-learn")
                
                # 趨勢分析
                if "📈 趨勢分析" in ai_features:
                    st.markdown("#### 📈 趨勢分析")
                    
                    trend_columns = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME', # universal_analysis_platform_v8_5_ai_ready.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime
import numpy as np

# 🚀 逐步引入AI功能的套件 (可選安裝)
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    AI_ANOMALY_AVAILABLE = True
except ImportError:
    AI_ANOMALY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# --- 版本資訊設定 ---
VERSION = "v8.5 AI-Ready"
VERSION_DATE = "2025年6月"
VERSION_FEATURES = [
    "🎨 保持v8.0美化界面設計",
    "🤖 新增AI異常檢測功能 (可選)",
    "📈 增強趨勢分析與預測",
    "🔍 智能數據洞察建議",
    "📊 交互式圖表支援 (Plotly)",
    "💬 基礎智能對話功能",
    "📝 智能報告生成器",
    "🎯 漸進式AI功能部署"
]

# --- AI功能模組 (輕量版) ---
class BasicAIEngine:
    """基礎AI引擎 - 不依賴複雜套件"""
    
    def __init__(self):
        self.analysis_history = []
    
    def basic_anomaly_detection(self, data_series, threshold_factor=2.5):
        """基礎異常檢測 - 使用統計方法"""
        if len(data_series) < 10:
            return None, "數據量不足"
        
        try:
            clean_data = pd.to_numeric(data_series, errors='coerce').dropna()
            if len(clean_data) < 5:
                return None, "有效數據點太少"
            
            mean = clean_data.mean()
            std = clean_data.std()
            
            # 使用統計方法檢測異常
            upper_bound = mean + threshold_factor * std
            lower_bound = mean - threshold_factor * std
            
            anomalies = clean_data[(clean_data > upper_bound) | (clean_data < lower_bound)]
            
            result = {
                'total_points': len(clean_data),
                'anomaly_count': len(anomalies),
                'anomaly_indices': anomalies.index.tolist(),
                'bounds': (lower_bound, upper_bound),
                'statistics': {'mean': mean, 'std': std}
            }
            
            return result, f"發現 {len(anomalies)} 個異常點"
            
        except Exception as e:
            return None, f"異常檢測失敗: {str(e)}"
    
    def advanced_anomaly_detection(self, df, columns=None):
        """進階異常檢測 - 使用機器學習 (需要scikit-learn)"""
        if not AI_ANOMALY_AVAILABLE:
            return None, "需要安裝 scikit-learn 套件"
        
        try:
            if columns is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                columns = [col for col in numeric_cols if col not in ['Date', 'sec', 'RT', 'TIME']]
            
            if not columns:
                return None, "無可用的數值欄位"
            
            data = df[columns].dropna()
            if len(data) < 10:
                return None, "數據量不足進行ML異常檢測"
            
            # 使用Isolation Forest
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(scaled_data)
            anomaly_scores = iso_forest.score_samples(scaled_data)
            
            result_df = df.copy()
            result_df['is_anomaly'] = False
            result_df.loc[data.index, 'is_anomaly'] = anomaly_labels == -1
            result_df.loc[data.index, 'anomaly_score'] = anomaly_scores
            
            anomaly_count = sum(anomaly_labels == -1)
            
            return result_df, f"ML檢測完成：發現 {anomaly_count} 個異常點"
            
        except Exception as e:
            return None, f"ML異常檢測失敗: {str(e)}"
    
    def trend_analysis(self, data_series, window_size=20):
        """趨勢分析"""
        try:
            clean_data = pd.to_numeric(data_series, errors='coerce').dropna()
            if len(clean_data) < window_size:
                return None, f"數據量不足（需要至少{window_size}個點）"
            
            # 計算移動平均
            rolling_mean = clean_data.rolling(window=window_size).mean()
            
            # 簡單線性趨勢
            x = np.arange(len(clean_data))
            if SCIPY_AVAILABLE:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_data)
                trend_strength = abs(r_value)
            else:
                # 簡單線性回歸
                slope = np.polyfit(x, clean_data, 1)[0]
                trend_strength = 0.5  # 預設值
                r_value = 0.5
            
            # 判斷趨勢方向
            if slope > 0.01:
                trend_direction = "上升"
            elif slope < -0.01:
                trend_direction = "下降"
            else:
                trend_direction = "穩定"
            
            result = {
                'slope': slope,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'rolling_mean': rolling_mean,
                'analysis': f"趨勢方向：{trend_direction}，強度：{'強' if trend_strength > 0.7 else '中' if trend_strength > 0.3 else '弱'}"
            }
            
            return result, "趨勢分析完成"
            
        except Exception as e:
            return None, f"趨勢分析失敗: {str(e)}"
    
    def generate_insights(self, df, analysis_results=None):
        """生成智能洞察"""
        insights = []
        
        if df is None or df.empty:
            return ["數據為空，無法生成洞察"]
        
        # 基本統計洞察
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append(f"📊 數據包含 {len(df)} 筆記錄和 {len(numeric_cols)} 個數值欄位")
            
            # 檢查數據變化程度
            for col in numeric_cols[:3]:  # 只分析前3個欄位
                data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(data) > 1:
                    cv = data.std() / data.mean() if data.mean() != 0 else 0
                    if cv > 0.5:
                        insights.append(f"🔍 {col} 變化程度較大 (變異係數: {cv:.2f})")
                    elif cv < 0.1:
                        insights.append(f"📈 {col} 相對穩定 (變異係數: {cv:.2f})")
        
        # 基於分析結果的洞察
        if analysis_results:
            for result in analysis_results:
                if "異常" in result:
                    insights.append("⚠️ 建議重點關注檢測到的異常點")
                if "趨勢" in result:
                    insights.append("📈 建議持續監控趨勢變化")
        
        # 通用建議
        insights.append("💡 建議定期執行異常檢測以維護數據品質")
        insights.append("🎯 可嘗試調整時間範圍來聚焦特定時段的分析")
        
        return insights

# --- 智能對話系統 (簡化版) ---
class SimpleConversationSystem:
    def __init__(self):
        self.responses = {
            'greeting': ['您好！我是您的數據分析助手，有什麼可以幫您的嗎？'],
            'analysis': ['我正在分析您的數據...', '讓我檢查一下數據模式...'],
            'anomaly': ['我發現了一些異常點，建議您仔細檢查', '數據中存在一些不尋常的模式'],
            'trend': ['根據數據趨勢分析...', '從時間序列來看...'],
            'help': ['您可以問我關於異常檢測、趨勢分析、或數據統計的問題']
        }
    
    def process_query(self, query, df_info=None):
        """處理用戶查詢"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['你好', 'hello', '嗨']):
            return self.responses['greeting'][0]
        elif any(word in query_lower for word in ['異常', 'anomaly', '問題']):
            return "建議執行異常檢測功能來識別數據中的異常模式。您可以在側邊欄選擇異常檢測選項。"
        elif any(word in query_lower for word in ['趨勢', 'trend', '預測']):
            return "我可以幫您分析數據趨勢。請在分析選項中選擇趨勢分析功能。"
        elif any(word in query_lower for word in ['最高', 'maximum', 'max']):
            return "請查看統計表格中的Tmax欄位，或使用圖表上的數據點提示功能。"
        elif any(word in query_lower for word in ['幫助', 'help', '怎麼用']):
            return self.responses['help'][0]
        else:
            return f"我理解您想了解：{query}。請嘗試使用更具體的關鍵詞，如'異常檢測'、'趨勢分析'、'最高溫度'等。"

# --- 創建交互式圖表 (如果有Plotly) ---
def create_interactive_chart(df, columns, title="Interactive Analysis"):
    """創建交互式圖表"""
    if not PLOTLY_AVAILABLE:
        return None
    
    if df is None or df.empty or not columns:
        return None
    
    fig = go.Figure()
    
    for col in columns[:5]:  # 限制最多5個series
        if col in df.columns:
            y_data = pd.to_numeric(df[col], errors='coerce').dropna()
            if not y_data.empty:
                fig.add_trace(go.Scatter(
                    x=df.index.total_seconds(),
                    y=y_data,
                    mode='lines',
                    name=col.replace('YOKO: ', '').replace('PTAT: ', ''),
                    line=dict(width=2),
                    hovertemplate=f'{col}<br>時間: %{{x}}s<br>數值: %{{y}}<extra></extra>'
                ))
    
    fig.update_layout(
        title=title,
        xaxis_title="時間 (秒)",
        yaxis_title="數值",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white'
    )
    
    return fig

# --- 子模組：PTAT Log 解析器 (修復版) ---
def parse_ptat(file_content):
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

# --- 子模組：YOKOGAWA Log 解析器 (靜默智能版) ---
def parse_yokogawa(file_content, is_excel=False):
    try:
        read_func = pd.read_excel if is_excel else pd.read_csv
        
        if is_excel:
            possible_headers = [29, 28, 30, 27]
        else:
            possible_headers = [0, 1, 2]
            
        df = None
        found_time_col = None
        successful_header = None
        
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
                        successful_header = header_row
                        break
                
                if found_time_col:
                    break
                    
            except Exception as e:
                continue
        
        if df is None or found_time_col is None:
            error_msg = "YOKOGAWA Log中找不到時間欄位。"
            if df is not None:
                error_msg += f" 可用欄位: {list(df.columns)[:15]}"
            return None, error_msg
        
        time_column = found_time_col
        
        if is_excel and successful_header == 29:
            try:
                file_content.seek(0)
                ch_row = pd.read_excel(file_content, header=None, skiprows=27, nrows=1).iloc[0]
                file_content.seek(0)
                tag_row = pd.read_excel(file_content, header=None, skiprows=28, nrows=1).iloc[0]
                
                new_column_names = {}
                for i, original_col in enumerate(df.columns):
                    if i < len(ch_row) and i < len(tag_row):
                        ch_name = str(ch_row.iloc[i]).strip() if pd.notna(ch_row.iloc[i]) else ""
                        tag_name = str(tag_row.iloc[i]).strip() if pd.notna(tag_row.iloc[i]) else ""
                        
                        if tag_name and tag_name != 'nan' and tag_name != 'Tag':
                            new_column_names[original_col] = tag_name
                        elif ch_name and ch_name != 'nan' and ch_name.startswith('CH'):
                            new_column_names[original_col] = ch_name
                        else:
                            new_column_names[original_col] = original_col
                    else:
                        new_column_names[original_col] = original_col
                
                df.rename(columns=new_column_names, inplace=True)
                
            except Exception as e:
                pass
        
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
                return None, f"無法解析時間格式 '{time_column}': {e}. 樣本: {time_series.head(3).tolist()}"
        
        valid_times_mask = df['time_index'].notna()
        if valid_times_mask.sum() == 0:
            return None, f"時間欄位 '{time_column}' 中沒有有效的時間數據"
        
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

# --- 主模組：解析器調度中心 (靜默版) ---
def parse_dispatcher(uploaded_file):
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
    
    try:
        if is_excel:
            try:
                file_content.seek(0)
                df_sniff = pd.read_excel(file_content, header=None, skiprows=27, nrows=1)
                
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
        
        else:
            try:
                file_content.seek(0)
                first_lines_text = "".join([file_content.readline().decode('utf-8', errors='ignore') for _ in range(10)])
                file_content.seek(0)
                
                if 'MSR Package Temperature' in first_lines_text:
                    return parse_ptat(file_content)
                else:
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

# --- PTAT Log 專用統計計算函式 ---
def calculate_ptat_stats(df, x_limits=None):
    """計算PTAT Log的專用統計數據"""
    if df is None or df.empty:
        return None, None, None
    
    df_stats = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_stats = df_stats[(df_stats.index >= x_min_td) & (df_stats.index <= x_max_td)]
    
    if df_stats.empty:
        return None, None, None
    
    freq_stats = []
    freq_cols = [col for col in df_stats.columns if 'frequency' in col.lower() and 'core' in col.lower()]
    
    lfm_value = "N/A"
    hfm_value = "N/A"
    
    for col in df_stats.columns:
        if 'lfm' in col.lower():
            lfm_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
            if len(lfm_data) > 0:
                lfm_value = f"{lfm_data.iloc[0]:.0f} MHz"
        elif 'hfm' in col.lower():
            hfm_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
            if len(hfm_data) > 0:
                hfm_value = f"{hfm_data.iloc[0]:.0f} MHz"
    
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

def generate
