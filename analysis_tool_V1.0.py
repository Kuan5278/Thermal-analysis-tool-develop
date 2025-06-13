# final_diagnostic_tool.py
import streamlit as st
import pandas as pd
import io

st.title("檔案結構最終檢視器")
st.info("這個工具將顯示您上傳檔案的前30行原始文字，這將是解決問題的最後一步。")

# 修正：同時支援 .csv 和 .xlsx
uploaded_file = st.sidebar.file_uploader("請上傳有問題的 Log File", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        st.success("檔案上傳成功！")
        file_content = io.BytesIO(uploaded_file.getvalue())
        is_excel = '.xlsx' in uploaded_file.name.lower()
        
        # 讀取前30行，不指定標頭
        read_func = pd.read_excel if is_excel else pd.read_csv
        df_raw = read_func(file_content, header=None, nrows=30)

        st.write("---")
        st.subheader("檔案前30行原始內容：")
        
        # 將DataFrame轉為純文字輸出
        output_text = df_raw.to_string()
        
        st.code(output_text, language='text')
        
        st.warning("請點擊上方區塊右上角的『複製』按鈕，然後將所有內容貼上給我。")

    except Exception as e:
        st.error(f"讀取檔案時發生錯誤: {e}")
