def generate_gpumon_chart(df, left_col, right_col, x_limits, y_limits=None):
    """生成GPUMon專用圖表"""
    if df is None or not left_col or left_col not in df.columns: 
        return None
    if right_col and right_col != 'None' and right_col not in df.columns: 
        return None
    
    df_chart = df.copy()
    if x_limits:
        x_min_td, x_max_td = pd.to_timedelta(x_limits[0], unit='s'), pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    if df_chart.empty: 
        return None
    
    df_chart.loc[:, 'left_val'] = pd.to_numeric(df_chart[left_col], errors='coerce')
    if right_col and right_col != 'None':
        df_chart.loc[:, 'right_val'] = pd.to_numeric(df_chart[right_col], errors='coerce')
    
    fig, ax1 = plt.subplots(figsize=(10.2, 5.1))
    
    title = f'GPUMon: {left_col.replace("GPU: ", "")} {"& " + right_col.replace("GPU: ", "") if right_col and right_col != "None" else ""}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    x_axis_seconds = df_chart.index.total_seconds()
    color = 'tab:orange'
    ax1.set_xlabel('Elapsed Time (seconds)', fontsize=11)
    ax1.set_ylabel(left_col.replace("GPU: ", ""), color=color, fontsize=11)
    ax1.plot(x_axis_seconds, df_chart['left_val'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    if y_limits:
        ax1.set_ylim(y_limits)
    
    if right_col and right_col != 'None':
        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel(right_col.replace("GPU: ", ""), color=color, fontsize=11)
        ax2.plot(x_axis_seconds, df_chart['right_val'], color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
    
    if x_limits: 
        ax1.set_xlim(x_limits)
    
    fig.tight_layout()
    return fig

def display_version_info():
    """顯示版本資訊"""
    with st.expander("📋 版本資訊", expanded=False):
        st.markdown(f"""
        **當前版本：{VERSION}** | **發布日期：{VERSION_DATE}**
        
        ### 🆕 本版本更新內容：
        - 🎨 全新美化界面設計
        - 📊 優化圖表大小與顯示比例
        - 📋 改進統計表格布局 (PTAT統計表格改為垂直排列)
        - 🔧 增強YOKOGAWA Excel智能解析
        - ⚡ 提升PTAT Log處理效能
        - 🎯 新增Y軸範圍自定義功能
        - 🛠️ 修復PTAT Log解析問題
        - ✨ PTAT統計表格垂直排列，無需滾動查看完整數據
        - 📈 YOKOGAWA圖表大小調整為與PTAT圖表一致
        - 🔄 YOKOGAWA統計數據移至圖表下方，布局更佳
        - 🚀 **新增GPUMon Log完整支持**
        - 🎮 **GPU溫度、功耗、頻率、使用率監控**
        - 📊 **GPU專用統計分析與視覺化**
        
        ---
        💡 **使用提示：** 支援YOKOGAWA Excel、PTAT CSV、GPUMon CSV格式，提供智能解析與多維度統計分析
        """)

def main():
    """主程式"""
    st.set_page_config(
        page_title="GPU & 溫度數據分析平台",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
        .gpumon-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
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
    
    st.markdown(f"""
    <div class="main-header">
        <h1>🎮 GPU & 溫度數據分析平台</h1>
        <p>智能解析 YOKOGAWA、PTAT、GPUMon Log 文件，提供專業級數據分析與視覺化</p>
        <p><strong>{VERSION}</strong> | {VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    display_version_info()
    
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True,
        help="支援 YOKOGAWA Excel、PTAT CSV、GPUMon CSV 格式"
    )
    
    if uploaded_files:
        st.sidebar.markdown("### 📂 已上傳檔案")
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / 1024
            st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
        
        st.sidebar.markdown("---")
        
        if len(uploaded_files) == 1:
            df_check, log_type_check = parse_dispatcher(uploaded_files[0])
            is_single_yokogawa = (log_type_check == "YOKOGAWA Log")
            is_single_gpumon = (log_type_check == "GPUMon Log")
        else:
            is_single_yokogawa = False
            is_single_gpumon = False
        
        if is_single_yokogawa:
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ 檔案解析成功</strong><br>
                📄 檔案類型：{log_type_check}<br>
                📊 數據筆數：{len(df_check):,} 筆<br>
                🔢 通道數量：{len([c for c in df_check.columns if df_check[c].dtype in ['float64', 'int64']]):,} 個
            </div>
            """, unsafe_allow_html=True)
            
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
            
            st.markdown("### 📈 YOKOGAWA 全通道溫度曲線圖")
            
            if df_check is not None:
                fig = generate_yokogawa_temp_chart(df_check, x_limits, y_limits)
                if fig: 
                    st.pyplot(fig, use_container_width=True)
                else: 
                    st.warning("⚠️ 無法產生溫度圖表")
            else:
                st.error("❌ 數據解析失敗")
            
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
        
        elif is_single_gpumon:
            st.markdown(f"""
            <div class="gpumon-box">
                <strong>🎮 GPUMon檔案解析成功</strong><br>
                📄 檔案類型：{log_type_check}<br>
                📊 數據筆數：{len(df_check):,} 筆<br>
                🔢 監控參數：{len([c for c in df_check.columns if df_check[c].dtype in ['float64', 'int64']]):,} 個
            </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown("### ⚙️ GPUMon 圖表設定")
            
            if len(df_check) > 0:
                x_min_val = df_check.index.min().total_seconds()
                x_max_val = df_check.index.max().total_seconds()
                
                if x_min_val < x_max_val:
                    x_min, x_max = st.sidebar.slider(
                        "⏱️ 時間範圍 (秒)", 
                        float(x_min_val), 
                        float(x_max_val), 
                        (float(x_min_val), float(x_max_val)),
                        key="gpumon_time_range"
                    )
                    x_limits = (x_min, x_max)
                else:
                    x_limits = None
            else:
                x_limits = None
            
            numeric_columns = df_check.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                st.sidebar.markdown("#### 🎯 參數選擇")
                
                temp_cols = [c for c in numeric_columns if 'Temperature' in c and 'GPU' in c]
                power_cols = [c for c in numeric_columns if 'Power' in c and ('NVVDD' in c or 'Total' in c)]
                freq_cols = [c for c in numeric_columns if 'Clock' in c and 'GPC' in c]
                util_cols = [c for c in numeric_columns if 'Utilization' in c and 'GPU' in c]
                
                default_left = (temp_cols[0] if temp_cols else 
                               power_cols[0] if power_cols else 
                               freq_cols[0] if freq_cols else 
                               numeric_columns[0])
                
                left_y_axis = st.sidebar.selectbox(
                    "📈 左側Y軸變數", 
                    options=numeric_columns, 
                    index=numeric_columns.index(default_left) if default_left in numeric_columns else 0
                )
                
                right_y_axis_options = ['None'] + numeric_columns
                default_right = 'None'
                
                if 'Temperature' in left_y_axis and power_cols:
                    default_right = power_cols[0]
                elif 'Power' in left_y_axis and temp_cols:
                    default_right = temp_cols[0]
                elif 'Clock' in left_y_axis and util_cols:
                    default_right = util_cols[0]
                    
                try: 
                    default_right_index = right_y_axis_options.index(default_right)
                except ValueError: 
                    default_right_index = 0
                    
                right_y_axis = st.sidebar.selectbox(
                    "📊 右側Y軸變數 (可選)", 
                    options=right_y_axis_options, 
                    index=default_right_index
                )
                
                st.sidebar.markdown("#### 🎚️ Y軸範圍")
                auto_y = st.sidebar.checkbox("🔄 自動Y軸範圍", value=True, key="gpumon_auto_y")
                y_limits = None
                
                if not auto_y and left_y_axis:
                    left_data = pd.to_numeric(df_check[left_y_axis], errors='coerce').dropna()
                    if len(left_data) > 0:
                        data_min, data_max = float(left_data.min()), float(left_data.max())
                        data_range = data_max - data_min
                        buffer = data_range * 0.1 if data_range > 0 else 1
                        y_min, y_max = st.sidebar.slider(
                            f"📊 {left_y_axis.replace('GPU: ', '')} 範圍",
                            data_min - buffer,
                            data_max + buffer,
                            (data_min - buffer, data_max + buffer),
                            step=0.1,
                            key="gpumon_y_range"
                        )
                        y_limits = (y_min, y_max)
                
                st.markdown("### 🎮 GPUMon 數據分析")
                
                fig = generate_gpumon_chart(df_check, left_y_axis, right_y_axis, x_limits, y_limits)
                if fig: 
                    st.pyplot(fig, use_container_width=True)
                    
                    st.markdown("### 📊 GPUMon 統計分析")
                    
                    temp_df, power_df, freq_df, util_df = calculate_gpumon_stats(df_check, x_limits)
                    
                    st.markdown("#### 🌡️ GPU溫度統計")
                    if temp_df is not None and not temp_df.empty:
                        st.dataframe(temp_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            ❓ 未找到GPU溫度數據
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("#### ⚡ GPU功耗統計")
                    if power_df is not None and not power_df.empty:
                        st.dataframe(power_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            ❓ 未找到GPU功耗數據
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("#### 🔄 GPU頻率統計")
                    if freq_df is not None and not freq_df.empty:
                        st.dataframe(freq_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            ❓ 未找到GPU頻率數據
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("#### 📊 GPU使用率統計")
                    if util_df is not None and not util_df.empty:
                        st.dataframe(util_df, use_container_width=True, hide_index=True)
                    else:
                        st.markdown("""
                        <div class="info-box">
                            ❓ 未找到GPU使用率數據
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    st.warning("⚠️ 無法產生GPUMon圖表")
            else:
                st.warning("⚠️ GPUMon Log中無可用的數值型數據")
        
        else:
            all_dfs = []
            log_types = []
            
            for file in uploaded_files:
                df, log_type = parse_dispatcher(file)
                if df is not None:
                    all_dfs.append(df)
                    log_types.append(log_type)
            
            if all_dfs:
                st.markdown("### 📋 檔案解析狀態")
                status_cols = st.columns(len(uploaded_files))
                
                for i, (file, log_type) in enumerate(zip(uploaded_files, log_types)):
                    with status_cols[i]:
                        if i < len(all_dfs):
                            if log_type == "GPUMon Log":
                                st.markdown(f"""
                                <div class="gpumon-box">
                                    <strong>🎮 {file.name}</strong><br>
                                    📄 {log_type}<br>
                                    📊 {len(all_dfs[i]):,} 筆數據
                                </div>
                                """, unsafe_allow_html=True)
                            else:
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
                
                has_ptat = any("PTAT" in log_type for log_type in log_types)
                has_gpumon = any("GPUMon" in log_type for log_type in log_types)
                
                if has_ptat and len(all_dfs) == 1:
                    ptat_df = all_dfs[0]
                    
                    st.sidebar.markdown("### ⚙️ PTAT 圖表設定")
                    
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
                        
                        st.markdown("### 🔬 PTAT Log 數據分析")
                        
                        fig = generate_flexible_chart(ptat_df, left_y_axis, right_y_axis, x_limits, y_limits)
                        if fig: 
                            st.pyplot(fig, use_container_width=True)
                            
                            st.markdown("### 📊 PTAT Log 統計分析")
                            
                            freq_df, power_df, temp_df = calculate_ptat_stats(ptat_df, x_limits)
                            
                            st.markdown("#### 🖥️ CPU Core Frequency")
                            if freq_df is not None and not freq_df.empty:
                                st.dataframe(freq_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("""
                                <div class="info-box">
                                    ❓ 未找到CPU頻率數據
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            st.markdown("#### ⚡ Package Power")
                            if power_df is not None and not power_df.empty:
                                st.dataframe(power_df, use_container_width=True, hide_index=True)
                            else:
                                st.markdown("""
                                <div class="info-box">
                                    ❓ 未找到Package Power數據
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            st.markdown("#### 🌡️ MSR Package Temperature")
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
                    master_df = pd.concat(all_dfs)
                    master_df_resampled = master_df.select_dtypes(include=['number']).resample('1S').mean(numeric_only=True).interpolate(method='linear')
                    numeric_columns = master_df_resampled.columns.tolist()

                    if numeric_columns:
                    if numeric_columns:
                        st.sidebar.markdown("### ⚙️ 圖表設定")
                        
                        gpu_temp_cols = [c for c in numeric_columns if 'GPU' in c and 'Temperature' in c]
                        cpu_temp_cols = [c for c in numeric_columns if 'PTAT' in c and 'Temp' in c]
                        yoko_temp_cols = [c for c in numeric_columns if 'YOKO' in c]
                        
                        default_left = (gpu_temp_cols[0] if gpu_temp_cols else 
                                       cpu_temp_cols[0] if cpu_temp_cols else 
                                       yoko_temp_cols[0] if yoko_temp_cols else 
                                       numeric_columns[0])
                        
                        left_y_axis = st.sidebar.selectbox(
                            "📈 左側Y軸變數", 
                            options=numeric_columns, 
                            index=numeric_columns.index(default_left) if default_left in numeric_columns else 0
                        )
                        
                        right_y_axis_options = ['None'] + numeric_columns
                        default_right_index = 0
                        
                        if len(numeric_columns) > 1:
                            gpu_power_cols = [c for c in numeric_columns if 'GPU' in c and 'Power' in c]
                            cpu_power_cols = [c for c in numeric_columns if 'PTAT' in c and 'Power' in c]
                            
                            if 'GPU' in left_y_axis and gpu_power_cols:
                                default_right = gpu_power_cols[0]
                            elif 'PTAT' in left_y_axis and cpu_power_cols:
                                default_right = cpu_power_cols[0]
                            else:
                                default_right = 'None'
                                
                            try: 
                                default_right_index = right_y_axis_options.index(default_right)
                            except ValueError: 
                                default_right_index = 1 if len(right_y_axis_options) > 1 else 0
                                
                        right_y_axis = st.sidebar.selectbox(
                            "📊 右側Y軸變數 (可選)", 
                            options=right_y_axis_options, 
                            index=default_right_index
                        )
                        
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
                        
                        st.markdown("### 🔀 混合數據分析圖表")
                        
                        source_summary = []
                        for log_type in set(log_types):
                            count = log_types.count(log_type)
                            if log_type == "GPUMon Log":
                                source_summary.append(f"🎮 {count}個GPUMon檔案")
                            elif log_type == "PTAT Log":
                                source_summary.append(f"🖥️ {count}個PTAT檔案")
                            elif log_type == "YOKOGAWA Log":
                                source_summary.append(f"📊 {count}個YOKOGAWA檔案")
                        
                        st.info(f"**數據來源：** {' + '.join(source_summary)}")
                        
                        if has_gpumon:
                            fig = generate_gpumon_chart(master_df_resampled, left_y_axis, right_y_axis, (x_min, x_max), y_limits)
                        else:
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
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件開始分析")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 支援格式")
            
            with st.container():
                st.success("📊 **YOKOGAWA Excel (.xlsx)**")
                st.caption("自動識別CH編號與Tag標籤")
            
            st.write("")
            
            with st.container():
                st.success("🖥️ **PTAT CSV (.csv)**")
                st.caption("CPU溫度、頻率、功耗分析")
            
            st.write("")
            
            with st.container():
                st.success("🎮 **GPUMon CSV (.csv)**")
                st.caption("GPU溫度、功耗、頻率、使用率")
        
        with col2:
            st.subheader("✨ 主要功能")
            
            st.write("🎯 **智能檔案格式識別**")
            st.caption("自動檢測並解析不同格式的Log檔案")
            
            st.write("📊 **即時數據統計分析**")
            st.caption("快速計算溫度統計和數據摘要")
            
            st.write("📈 **動態圖表與範圍調整**") 
            st.caption("可調整時間和數值範圍的互動圖表")
            
            st.write("🔄 **多檔案混合比較**")
            st.caption("同時分析多個檔案並進行比較")
            
            st.write("🎮 **GPU專業監控**")
            st.caption("GPU溫度、功耗、頻率全方位分析")
        
        with col3:
            st.subheader("🎮 GPU監控特色")
            
            st.write("🌡️ **多點溫度監控**")
            st.caption("GPU核心與顯存溫度分離監控")
            
            st.write("⚡ **精確功耗分析**")
            st.caption("NVVDD、FBVDD、MSVDD分軌監控")
            
            st.write("🔄 **動態頻率追蹤**")
            st.caption("GPC與Memory頻率即時監控")
            
            st.write("📊 **使用率統計**")
            st.caption("GPU、FB、Video使用率分析")
            
            st.write("🏷️ **狀態監控**")
            st.caption("P-State、限制原因智能分析")
        
        st.divider()
        
        st.subheader("📖 快速使用指南")
        
        step_col1, step_col2, step_col3, step_col4 = st.columns(4)
        
        with step_col1:
            st.info("""
            **步驟 1: 上傳檔案**  
            點擊左側的檔案上傳區域，選擇您的Log檔案
            """)
        
        with step_col2:
            st.info("""
            **步驟 2: 自動解析**  
            系統會智能識別檔案格式並自動解析數據內容
            """)
        
        with step_col3:
            st.info("""
            **步驟 3: 設定參數**  
            在側邊欄選擇要分析的參數和時間範圍
            """)
        
        with step_col4:
            st.info("""
            **步驟 4: 查看結果**  
            在圖表和統計表格中查看分析結果
            """)
        
        st.subheader("🎮 GPUMon 監控能力")
        
        gpu_col1, gpu_col2 = st.columns(2)
        
        with gpu_col1:
            st.markdown("""
            **🔥 溫度監控範圍：**
            - GPU核心溫度
            - 顯存溫度  
            - CPU溫度
            - 16個平台溫度感測器
            
            **⚡ 功耗監控精度：**
            - NVVDD功耗軌
            - FBVDD功耗軌
            - MSVDD功耗軌
            - 系統總功耗
            """)
        
        with gpu_col2:
            st.markdown("""
            **🔄 性能監控項目：**
            - GPC核心頻率
            - 顯存頻率
            - GPU使用率
            - 顯存使用率
            - 視頻編解碼使用率
            
            **🏷️ 狀態監控功能：**
            - P-State電源狀態
            - 限制原因分析
            - RTD3/GC6狀態
            """)
        
        st.warning("💡 **需要幫助？** 如有任何問題，請聯繫技術支援團隊")
        
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
        - [🎮 GPUMon指南](https://example.com/gpumon-guide)
        """)
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
        🎮 GPU & 溫度數據分析平台 {VERSION} | 由 Streamlit 驅動 | © 2025 [您的公司名稱] 版權所有
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                # universal_analysis_platform_v9_0_gpumon.py
# 完整的數據分析平台 - 新增GPUMon支持

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime
import numpy as np

# 版本資訊
VERSION = "v9.0 GPUMon"
VERSION_DATE = "2025年6月"

# --- 子模組：GPUMon Log 解析器 (全新) ---
def parse_gpumon(file_content):
    """GPUMon Log解析器"""
    try:
        file_content.seek(0)
        content = file_content.read().decode('utf-8', errors='ignore')
        lines = content.split('\n')
        
        # 尋找數據標題行 (包含 "Iteration, Date, Timestamp")
        header_row_index = None
        for i, line in enumerate(lines):
            if 'Iteration' in line and 'Date' in line and 'Timestamp' in line:
                header_row_index = i
                break
        
        if header_row_index is None:
            return None, "GPUMon Log中找不到數據標題行"
        
        # 解析標題
        header_line = lines[header_row_index]
        headers = [h.strip() for h in header_line.split(',')]
        
        # 解析數據行
        data_rows = []
        for i in range(header_row_index + 1, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith(','):
                try:
                    row_data = [cell.strip() for cell in line.split(',')]
                    if len(row_data) >= 3 and row_data[0].isdigit():  # 確保有Iteration數據
                        data_rows.append(row_data)
                except:
                    continue
        
        if not data_rows:
            return None, "GPUMon Log中沒有有效的數據行"
        
        # 創建DataFrame
        max_cols = max(len(headers), max(len(row) for row in data_rows))
        
        # 補齊標題
        while len(headers) < max_cols:
            headers.append(f'Column_{len(headers)}')
        
        # 補齊數據行
        for row in data_rows:
            while len(row) < max_cols:
                row.append('')
        
        df = pd.DataFrame(data_rows, columns=headers[:max_cols])
        
        # 處理時間數據
        try:
            # 合併日期和時間
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], errors='coerce')
            
            # 創建時間索引
            df['time_index'] = df['DateTime'] - df['DateTime'].iloc[0]
            
            # 過濾有效時間的行
            valid_mask = df['time_index'].notna()
            df = df[valid_mask].copy()
            
        except Exception as e:
            return None, f"GPUMon Log時間解析失敗: {e}"
        
        if df.empty:
            return None, "GPUMon Log時間解析後無有效數據"
        
        # 數值型欄位轉換
        numeric_columns = []
        for col in df.columns:
            if col not in ['Date', 'Timestamp', 'DateTime', 'time_index', 'Iteration']:
                # 嘗試轉換為數值
                try:
                    # 處理特殊值
                    df[col] = df[col].replace(['N/A', 'n/a', '', ' '], np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_columns.append(col)
                except:
                    pass
        
        # 添加前綴標識
        df = df.add_prefix('GPU: ')
        df.rename(columns={'GPU: time_index': 'time_index'}, inplace=True)
        
        return df.set_index('time_index'), "GPUMon Log"
        
    except Exception as e:
        return None, f"解析GPUMon Log時出錯: {e}"

# --- 子模組：PTAT Log 解析器 (修復版) ---
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

# --- 子模組：YOKOGAWA Log 解析器 ---
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
        
        # 智能欄位重命名功能
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
        
        # 處理時間數據
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

# --- 主模組：解析器調度中心 (更新版) ---
def parse_dispatcher(uploaded_file):
    """解析器調度中心 - 支援GPUMon"""
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
    
    try:
        if is_excel:
            # Excel檔案 - YOKOGAWA格式
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
            # CSV檔案 - 需要識別PTAT或GPUMon
            try:
                file_content.seek(0)
                # 讀取前面多行來識別格式
                first_content = ""
                for _ in range(50):  # 讀取前50行
                    try:
                        line = file_content.readline().decode('utf-8', errors='ignore')
                        if not line:
                            break
                        first_content += line
                    except:
                        break
                
                file_content.seek(0)
                
                # GPUMon格式識別
                if ('GPU Informations' in first_content or 
                    'Iteration, Date, Timestamp' in first_content or
                    'Temperature GPU (C)' in first_content or
                    'NVIDIA Graphics Device' in first_content):
                    return parse_gpumon(file_content)
                
                # PTAT格式識別  
                elif ('MSR Package Temperature' in first_content or 
                      'Version,Date,Time' in first_content):
                    return parse_ptat(file_content)
                
                # 預設嘗試YOKOGAWA格式
                else:
                    return parse_yokogawa(file_content, is_excel)
                    
            except Exception as e:
                # 最後嘗試PTAT格式
                file_content.seek(0)
                return parse_ptat(file_content)
        
    except Exception as e:
        pass
        
    return None, f"未知的Log檔案格式: {filename}"

# --- GPUMon專用統計計算函式 ---
def calculate_gpumon_stats(df, x_limits=None):
    """計算GPUMon Log的專用統計數據"""
    if df is None or df.empty:
        return None, None, None, None
    
    df_stats = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_stats = df_stats[(df_stats.index >= x_min_td) & (df_stats.index <= x_max_td)]
    
    if df_stats.empty:
        return None, None, None, None
    
    # GPU溫度統計
    temp_stats = []
    temp_cols = [col for col in df_stats.columns if 'Temperature' in col and 'GPU' in col]
    
    for col in temp_cols:
        temp_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(temp_data) > 0:
            temp_stats.append({
                'Temperature Sensor': col.replace('GPU: ', ''),
                'Max (°C)': f"{temp_data.max():.2f}",
                'Min (°C)': f"{temp_data.min():.2f}",
                'Avg (°C)': f"{temp_data.mean():.2f}"
            })
    
    temp_df = pd.DataFrame(temp_stats) if temp_stats else None
    
    # GPU功耗統計
    power_stats = []
    power_cols = [col for col in df_stats.columns if 'Power' in col and any(x in col for x in ['NVVDD', 'FBVDD', 'MSVDD', 'Total System'])]
    
    for col in power_cols:
        power_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(power_data) > 0:
            power_stats.append({
                'Power Rail': col.replace('GPU: ', ''),
                'Max (W)': f"{power_data.max():.2f}",
                'Min (W)': f"{power_data.min():.2f}",
                'Avg (W)': f"{power_data.mean():.2f}"
            })
    
    power_df = pd.DataFrame(power_stats) if power_stats else None
    
    # GPU頻率統計
    freq_stats = []
    freq_cols = [col for col in df_stats.columns if 'Clock' in col and any(x in col for x in ['GPC', 'Memory'])]
    
    for col in freq_cols:
        freq_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(freq_data) > 0:
            freq_stats.append({
                'Clock Domain': col.replace('GPU: ', ''),
                'Max (MHz)': f"{freq_data.max():.0f}",
                'Min (MHz)': f"{freq_data.min():.0f}",
                'Avg (MHz)': f"{freq_data.mean():.0f}"
            })
    
    freq_df = pd.DataFrame(freq_stats) if freq_stats else None
    
    # GPU使用率統計
    util_stats = []
    util_cols = [col for col in df_stats.columns if 'Utilization' in col]
    
    for col in util_cols:
        util_data = pd.to_numeric(df_stats[col], errors='coerce').dropna()
        if len(util_data) > 0:
            util_stats.append({
                'Utilization Type': col.replace('GPU: ', ''),
                'Max (%)': f"{util_data.max():.1f}",
                'Min (%)': f"{util_data.min():.1f}",
                'Avg (%)': f"{util_data.mean():.1f}"
            })
    
    util_df = pd.DataFrame(util_stats) if util_stats else None
    
    return temp_df, power_df, freq_df, util_df

# --- 溫度統計計算函式 ---
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
    
    # CPU Core Frequency 統計
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
    
    # Package Power 統計
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
    
    # MSR Package Temperature 統計
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

# --- 圖表繪製函式 ---
def generate_yokogawa_temp_chart(df, x_limits=None, y_limits=None):
    """改進版YOKOGAWA溫度圖表"""
    if df is None: 
        return None
    
    df_chart = df.copy()
    if x_limits:
        x_min_td = pd.to_timedelta(x_limits[0], unit='s')
        x_max_td = pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    
    if df_chart.empty:
        return None
    
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    
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

def generate_flexible_chart(df, left_col, right_col, x_limits, y_limits=None):
    """生成靈活的雙軸圖表"""
    if df is None or not left_col or left_col not in df.columns: 
        return None
    if right_col and right_col != 'None' and right_col not in df.columns: 
        return None
    
    df_chart = df.copy()
    if x_limits:
        x_min_td, x_max_td = pd.to_timedelta(x_limits[0], unit='s'), pd.to_timedelta(x_limits[1], unit='s')
        df_chart = df_chart[(df_chart.index >= x_min_td) & (df_chart.index <= x_max_td)]
    if df_chart.empty: 
        return None
    
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

def generate_gpumon_chart(df, left_col, right_col, x_limits, y_limits=None):
    """生
