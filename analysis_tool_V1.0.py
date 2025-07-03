def main():
    """主程式 - v10.3.9 Multi-File Analysis with Summary (Simplified)"""
    st.set_page_config(
        page_title="溫度數據視覺化平台",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS樣式
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
        .stMetric {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border: 1px solid #dee2e6;
        }
        .temp-summary-table {
            font-size: 0.9em;
        }
        .temp-summary-table th {
            background-color: #f0f2f6;
            font-weight: bold;
            text-align: center;
        }
        .temp-summary-table td {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # 標題
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 溫度數據視覺化平台</h1>
        <p>智能解析 YOKOGAWA、PTAT、GPUMon、System Log 文件 | 多檔案獨立分析 + Summary整合</p>
        <p><strong>{VERSION}</strong> | {VERSION_DATE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    display_version_info()
    
    # 初始化解析器註冊系統
    parser_registry = ParserRegistry()
    parser_registry.register(GPUMonParser())
    parser_registry.register(PTATParser())
    parser_registry.register(SystemLogParser()) # 註冊新的解析器
    parser_registry.register(YokogawaParser())  # 兜底解析器
    
    # 側邊欄
    st.sidebar.markdown("### 🎛️ 控制面板")
    st.sidebar.markdown("---")
    
    # =========================================================================
    # START: 修改此處以接受 .txt 文件
    # =========================================================================
    uploaded_files = st.sidebar.file_uploader(
        "📁 上傳Log File (可多選)", 
        type=['csv', 'xlsx', 'txt'], # <--- 在此加入 'txt'
        accept_multiple_files=True,
        help="v10.3.9 版：支援 .csv, .xlsx, .txt 格式的日誌檔案"
    )
    # =========================================================================
    # END: 修改完成
    # =========================================================================

    # 顯示訪問計數器
    display_visit_counter()
    
    if uploaded_files:
        # 顯示上傳檔案資訊
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📂 已上傳檔案")
        for i, file in enumerate(uploaded_files, 1):
            file_size = len(file.getvalue()) / 1024
            st.sidebar.markdown(f"**{i}.** `{file.name}` ({file_size:.1f} KB)")
        
        st.sidebar.markdown("---")
        
        # 解析檔案 - 超簡潔版本
        log_data_list = []
        for uploaded_file in uploaded_files:
            log_data = parser_registry.parse_file(uploaded_file)
            if log_data:
                log_data_list.append(log_data)
        
        if not log_data_list:
            st.error("❌ 無法解析任何檔案")
            return
        
        # 根據檔案數量決定UI模式
        if len(log_data_list) == 1:
            # 單檔案模式
            log_data = log_data_list[0]
            renderer = RendererFactory.create_renderer(log_data)
            
            if renderer:
                renderer.render(file_index=0)
            else:
                st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
        
        else:
            # 多檔案模式 - 每個檔案獨立顯示 + Summary整合
            st.success(f"📊 多檔案分析模式：成功解析 {len(log_data_list)} 個檔案")
            
            # 創建標籤頁，每個檔案一個標籤 + Summary標籤
            tab_names = []
            
            # 首先添加Summary標籤
            tab_names.append("📋 Summary")
            
            # 然後添加各個檔案的標籤
            for i, log_data in enumerate(log_data_list):
                # 生成標籤名稱
                filename = log_data.metadata.filename
                log_type = log_data.metadata.log_type
                
                # 縮短檔案名稱以適應標籤顯示
                short_name = filename
                if len(filename) > 15:
                    name_parts = filename.split('.')
                    if len(name_parts) > 1:
                        short_name = name_parts[0][:12] + "..." + name_parts[-1]
                    else:
                        short_name = filename[:12] + "..."
                
                # 添加類型emoji
                if "GPUMon" in log_type:
                    tab_name = f"🎮 {short_name}"
                elif "PTAT" in log_type:
                    tab_name = f"🖥️ {short_name}"
                elif "System Log" in log_type:
                    tab_name = f"📝 {short_name}" # 新增的日誌類型圖標
                elif "YOKOGAWA" in log_type:
                    tab_name = f"📊 {short_name}"
                else:
                    tab_name = f"📄 {short_name}"
                
                tab_names.append(tab_name)
            
            # 創建標籤頁
            tabs = st.tabs(tab_names)
            
            # 首先渲染Summary標籤頁
            with tabs[0]:
                summary_renderer = SummaryRenderer(log_data_list)
                summary_renderer.render()
            
            # 然後為每個檔案渲染獨立的內容
            for i, (tab, log_data) in enumerate(zip(tabs[1:], log_data_list)):
                with tab:
                    # 顯示檔案資訊
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #1f77b4;">
                        <h4>📁 檔案資訊</h4>
                        <p><strong>檔案名稱：</strong> {log_data.metadata.filename}</p>
                        <p><strong>檔案類型：</strong> {log_data.metadata.log_type}</p>
                        <p><strong>數據規模：</strong> {log_data.metadata.rows} 行 × {log_data.metadata.columns} 列</p>
                        <p><strong>檔案大小：</strong> {log_data.metadata.file_size_kb:.1f} KB</p>
                        <p><strong>時間範圍：</strong> {log_data.metadata.time_range}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 為每個檔案創建獨立的渲染器
                    renderer = RendererFactory.create_renderer(log_data)
                    
                    if renderer:
                        # 渲染該檔案的完整UI，傳遞正確的file_index
                        renderer.render(file_index=i)
                        
                    else:
                        st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
                        
                        # 顯示基本信息作為備用
                        st.markdown("### 📊 基本數據預覽")
                        if not log_data.df.empty:
                            st.write("**欄位列表：**")
                            for col in log_data.df.columns:
                                st.write(f"- {col}")
                            
                            st.write("**數據樣本（前5行）：**")
                            st.dataframe(log_data.df.head(), use_container_width=True)
            
            # 在標籤頁外提供檔案選擇器（用於側邊欄控制）
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🎛️ 多檔案控制")
            
            selected_file_index = st.sidebar.selectbox(
                "選擇要控制的檔案",
                options=range(len(log_data_list)),
                format_func=lambda x: f"{log_data_list[x].metadata.filename} ({log_data_list[x].metadata.log_type})",
                help="選擇要在側邊欄中控制的檔案"
            )
            
            st.sidebar.info(f"💡 當前選擇：{log_data_list[selected_file_index].metadata.filename}")
            # 注意：這個選擇器主要用於顯示信息，實際的控制是在各個tab中獨立進行的
    
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 Log 文件進行分析")
        
        st.markdown("""
        ### 📋 支援的檔案格式
        
        - **🎮 GPUMon CSV** - GPU性能監控數據（溫度、功耗、頻率、使用率）
        - **🖥️ PTAT CSV** - CPU性能監控數據（頻率、功耗、溫度）
        - **📝 System Log TXT** - (新增) CPU/GPU多核心詳細日誌
        - **📊 YOKOGAWA Excel/CSV** - 多通道溫度記錄儀數據
        
        ### ✨ 主要功能
        
        - **📋 智能解析** - 自動識別不同類型的Log檔案格式
        - **🎯 多檔案分析** - 同時上傳多個檔案，每個檔案獨立分析
        - **📊 即時互動** - 時間範圍和參數調整即時更新圖表
        - **📋 Summary整合** - 所有溫度數據整合成帶邊框HTML表格
        - **💾 一鍵複製** - HTML表格可直接複製到Word保留格式
        
        ### 🎯 使用流程
        
        1. **上傳檔案** - 在左側選擇一個或多個Log檔案
        2. **查看分析** - 每個檔案都有專屬的標籤頁和圖表控制
        3. **整合報告** - 在Summary標籤頁查看所有溫度數據整合表格
        4. **複製使用** - 直接複製HTML表格到Word或Excel
        """)


if __name__ == "__main__":
    main()

