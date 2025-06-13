# ultimate_diagnostic_tool.py
import streamlit as st
import pandas as pd
import io

st.set_page_config(layout="wide")
st.title("最終標頭健檢工具")
st.info("這個工具將會解析您檔案的複雜標頭結構，並顯示程式最終組合出的欄位名稱列表。")

uploaded_file = st.sidebar.file_uploader("請上傳 YOKOGAWA XLSX/CSV 檔案", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        is_excel = '.xlsx' in uploaded_file.name.lower()
        read_func = pd.read_excel if is_excel else pd.read_csv
        file_content = io.BytesIO(uploaded_file.getvalue())

        st.subheader("程式解析步驟與結果：")

        # 1. 讀取第28行 (CHXXX)
        ch_row = read_func(file_content, header=None, skiprows=27, nrows=1).iloc[0].astype(str).str.strip()
        st.write("1. 讀取第28行 (CHXXX)... 成功。")
        file_content.seek(0)
        
        # 2. 讀取第29行 (使用者Tag)
        tag_row = read_func(file_content, header=None, skiprows=28, nrows=1).iloc[0].astype(str).str.strip()
        st.write("2. 讀取第29行 (使用者Tag)... 成功。")
        file_content.seek(0)

        # 3. 讀取第30行 (主要標頭)
        main_header_row = read_func(file_content, header=None, skiprows=29, nrows=1).iloc[0].astype(str).str.strip()
        st.write("3. 讀取第30行 (主要標頭)... 成功。")
        
        # 4. 智慧組合最終的標頭
        final_columns = []
        final_columns.extend(main_header_row.tolist()[:3]) 
        for i in range(3, len(main_header_row)):
            tag_name = tag_row[i] if i < len(tag_row) else None
            ch_name = ch_row[i] if i < len(ch_row) else f'Fallback_{i}'
            
            if pd.notna(tag_name) and tag_name.lower() != 'nan' and tag_name != '':
                final_columns.append(tag_name)
            else:
                final_columns.append(ch_name)
        st.write("4. 智慧組合三行標頭... 成功。")

        # 5. 顯示最終結果
        st.write("---")
        st.success("程式最終組合出的欄位名稱列表如下：")
        st.code(final_columns)
        
        # 6. 檢查 'Time' 是否存在
        if 'Time' in final_columns:
            st.success("✅ 在最終的欄位列表**有**找到 'Time'。")
        else:
            st.error("❌ 在最終的欄位列表**沒有**找到 'Time'。這就是問題的根源！")
            
        st.warning("請將上面這個列表完整複製給我，我保證這是解決問題所需的最後一項資訊。")

    except Exception as e:
        st.error(f"在嘗試讀取標頭時發生錯誤: {e}")
