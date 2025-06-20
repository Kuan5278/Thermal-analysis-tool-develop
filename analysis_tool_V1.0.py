# 根據檔案數量決定UI模式
        if len(log_data_list) == 1:
            # 單檔案模式
            log_data = log_data_list[0]
            renderer = RendererFactory.create_renderer(log_data)
            
            if renderer:
                renderer.render()
            else:
                st.error(f"不支援的Log類型: {log_data.metadata.log_type}")
        
        else:
            # 多檔案模式 - 每個檔案獨立顯示
            st.success(f"📊 多檔案分析模式：成功解析 {len(log_data_list)} 個檔案")
            
            # 創建標籤頁，每個檔案一個標籤
            tab_names = []
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
                elif "YOKOGAWA" in log_type:
                    tab_name = f"📊 {short_name}"
                else:
                    tab_name = f"📄 {short_name}"
                
                tab_names.append(tab_name)
            
            # 創建標籤頁
            tabs = st.tabs(tab_names)
            
            # 為每個檔案渲染獨立的內容
            for i, (tab, log_data) in enumerate(zip(tabs, log_data_list)):
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
                        # 修改sidebar key以避免衝突
                        original_sidebar_key_suffix = getattr(st.session_state, 'current_file_index', 0)
                        st.session_state.current_file_index = i
                        
                        # 渲染該檔案的完整UI
                        renderer.render()
                        
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
            
            st.sidebar.info(f"💡 當前控制：{log_data_list[selected_file_index].metadata.filename}")
            st.session_state.current_file_index = selected_file_index
