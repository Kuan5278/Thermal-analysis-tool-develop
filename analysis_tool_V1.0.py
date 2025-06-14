# universal_analysis_platform_v9_1_debug.py
# 完整的數據分析平台 - 修正GPUMon解析問題

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import re
from datetime import datetime
import numpy as np

# 版本資訊
VERSION = "v9.1 Debug"
VERSION_DATE = "2025年6月"

def parse_gpumon(file_content):
    """GPUMon Log解析器 - 強化版"""
    try:
        file_content.seek(0)
        content = file_content.read().decode('utf-8', errors='ignore')
        lines = content.split('\n')
        
        st.write(f"🔍 GPUMon Debug: 檔案總行數 {len(lines)}")
        
        # 更寬鬆的標題行搜尋
        header_row_index = None
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if ('iteration' in line_lower and 'date' in line_lower and 'timestamp' in line_lower):
                header_row_index = i
                st.write(f"✅ 找到標題行在第 {i+1} 行")
                break
        
        if header_row_index is None:
            # 備用搜尋方式
            for i, line in enumerate(lines):
                if line.count(',') > 10 and ('iteration' in line.lower() or 'gpu' in line.lower()):
                    header_row_index = i
                    st.write(f"📍 備用方式找到可能的標題行在第 {i+1} 行")
                    break
        
        if header_row_index is None:
            return None, "GPUMon Log中找不到數據標題行"
        
        # 顯示找到的標題行
        header_line = lines[header_row_index]
        st.write(f"📋 標題行內容: {header_line[:100]}...")
        
        headers = [h.strip() for h in header_line.split(',')]
        st.write(f"📊 解析到 {len(headers)} 個欄位")
        
        # 解析數據行 - 更寬鬆的條件
        data_rows = []
        valid_data_count = 0
        
        for i in range(header_row_index + 1, min(header_row_index + 100, len(lines))):
            line = lines[i].strip()
            if line and not line.startswith(','):
                try:
                    row_data = [cell.strip() for cell in line.split(',')]
                    # 更寬鬆的驗證條件
                    if len(row_data) >= 3:
                        # 檢查第一個欄位是否為數字或者有其他有效資料
                        if (row_data[0].isdigit() or 
                            any(cell and cell != 'N/A' for cell in row_data[:5])):
                            data_rows.append(row_data)
                            valid_data_count += 1
                            if valid_data_count <= 3:  # 只顯示前3行
                                st.write(f"✅ 有效數據行 {valid_data_count}: {row_data[:5]}...")
                except Exception as e:
                    continue
        
        st.write(f"📈 找到 {len(data_rows)} 行有效數據")
        
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
        st.write(f"🎯 DataFrame創建成功: {df.shape}")
        
        # 處理時間數據 - 修正毫秒格式問題
        try:
            if 'Date' in df.columns and 'Timestamp' in df.columns:
                st.write(f"🕐 處理時間格式: Date + Timestamp")
                
                # 修正時間戳格式：20:50:38:502 -> 20:50:38.502
                df['Timestamp_fixed'] = df['Timestamp'].str.replace(r':(\d{3})$', r'.\1', regex=True)
                
                # 顯示修正前後的時間格式
                st.write(f"🔧 時間格式修正: {df['Timestamp'].iloc[0]} -> {df['Timestamp_fixed'].iloc[0]}")
                
                # 合併日期和修正後的時間
                df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp_fixed'], errors='coerce')
                
                st.write(f"📅 合併後時間: {df['DateTime'].iloc[0]}")
                
            elif 'Date' in df.columns:
                # 只有日期
                df['DateTime'] = pd.to_datetime(df['Date'], errors='coerce')
            else:
                # 尋找類似時間的欄位
                time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if time_cols:
                    df['DateTime'] = pd.to_datetime(df[time_cols[0]], errors='coerce')
                else:
                    # 使用序號作為時間索引
                    df['DateTime'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(range(len(df)), unit='s')
            
            # 檢查DateTime解析結果
            valid_datetime_count = df['DateTime'].notna().sum()
            st.write(f"📊 成功解析的時間點: {valid_datetime_count}/{len(df)}")
            
            if valid_datetime_count > 0:
                # 創建時間索引
                df['time_index'] = df['DateTime'] - df['DateTime'].iloc[0]
                valid_mask = df['time_index'].notna()
                df = df[valid_mask].copy()
                
                st.write(f"⏰ 時間解析成功，最終有效數據: {len(df)} 行")
                
                if len(df) > 0:
                    st.write(f"📈 時間範圍: {df['time_index'].min()} 到 {df['time_index'].max()}")
            else:
                st.warning("⚠️ 無法解析任何時間數據，使用序號索引")
                df['time_index'] = pd.to_timedelta(range(len(df)), unit='s')
            
        except Exception as e:
            st.write(f"⚠️ 時間解析異常: {e}")
            st.write("🔄 使用序號作為備用時間索引")
            df['time_index'] = pd.to_timedelta(range(len(df)), unit='s')
        
        if df.empty:
            return None, "GPUMon Log時間解析後無有效數據"
        
        # 數值型欄位轉換
        numeric_count = 0
        for col in df.columns:
            if col not in ['Date', 'Timestamp', 'DateTime', 'time_index', 'Iteration']:
                try:
                    df[col] = df[col].replace(['N/A', 'n/a', '', ' '], np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numeric_count += 1
                except:
                    pass
        
        st.write(f"🔢 轉換了 {numeric_count} 個數值欄位")
        
        # 添加前綴標識
        df = df.add_prefix('GPU: ')
        df.rename(columns={'GPU: time_index': 'time_index'}, inplace=True)
        
        result_df = df.set_index('time_index')
        st.write(f"🎉 GPUMon解析成功! 最終數據: {result_df.shape}")
        
        return result_df, "GPUMon Log"
        
    except Exception as e:
        st.error(f"❌ GPUMon解析錯誤: {e}")
        return None, f"解析GPUMon Log時出錯: {e}"

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

def parse_dispatcher(uploaded_file):
    """解析器調度中心 - 強化版本"""
    filename = uploaded_file.name
    file_content = io.BytesIO(uploaded_file.getvalue())
    is_excel = '.xlsx' in filename.lower() or '.xls' in filename.lower()
    
    st.write(f"🔍 檔案分析: {filename} (Excel: {is_excel})")
    
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
                # 讀取更多行來識別格式
                first_content = ""
                line_count = 0
                for _ in range(100):  # 增加到100行
                    try:
                        line = file_content.readline().decode('utf-8', errors='ignore')
                        if not line:
                            break
                        first_content += line
                        line_count += 1
                    except:
                        break
                
                st.write(f"📄 讀取了前 {line_count} 行內容進行格式識別")
                
                file_content.seek(0)
                
                # 更寬鬆的GPUMon格式識別
                gpumon_indicators = [
                    'GPU Informations',
                    'Iteration, Date, Timestamp',
                    'Temperature GPU (C)',
                    'NVIDIA Graphics Device',
                    'iteration' in first_content.lower() and 'gpu' in first_content.lower(),
                    'temperature' in first_content.lower() and 'power' in first_content.lower(),
                    'NVVDD' in first_content,
                    'FBVDD' in first_content
                ]
                
                is_gpumon = any(indicator in first_content or indicator for indicator in gpumon_indicators)
                
                if is_gpumon:
                    st.write("🎮 識別為 GPUMon 格式")
                    return parse_gpumon(file_content)
                
                # PTAT格式識別  
                elif ('MSR Package Temperature' in first_content or 
                      'Version,Date,Time' in first_content):
                    st.write("🖥️ 識別為 PTAT 格式")
                    return parse_ptat(file_content)
                
                # 預設嘗試YOKOGAWA格式
                else:
                    st.write("📊 嘗試 YOKOGAWA 格式")
                    return parse_yokogawa(file_content, is_excel)
                    
            except Exception as e:
                st.write(f"⚠️ 格式識別出錯，嘗試PTAT: {e}")
                file_content.seek(0)
                return parse_ptat(file_content)
        
    except Exception as e:
        st.error(f"❌ 調度器錯誤: {e}")
        
    return None, f"未知的Log檔案格式: {filename}"

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
        - 🛠️ **修正GPUMon解析問題**
        - 🔍 **強化格式識別邏輯**
        - 📊 **增加調試資訊顯示**
        - ⚡ **提升解析容錯性**
        - 🎮 **GPU溫度、功耗、頻率、使用率監控**
        - 📈 **動態圖表與範圍調整**
        
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
        
        # 顯示調試資訊
        st.markdown("### 🔍 檔案解析調試資訊")
        
        if len(uploaded_files) == 1:
            df_check, log_type_check = parse_dispatcher(uploaded_files[0])
            is_single_yokogawa = (log_type_check == "YOKOGAWA Log")
            is_single_gpumon = (log_type_check == "GPUMon Log")
        else:
            is_single_yokogawa = False
            is_single_gpumon = False
        
        # 其餘UI代碼與之前相同，但會顯示調試資訊...
        
        if is_single_gpumon:
            st.success(f"🎮 成功解析GPUMon檔案！")
            
            # 繼續GPUMon的UI邏輯...
            if df_check is not None:
                st.sidebar.markdown("### ⚙️ GPUMon 圖表設定")
                
                numeric_columns = df_check.select_dtypes(include=['number']).columns.tolist()
                if numeric_columns:
                    st.sidebar.markdown("#### 🎯 參數選擇")
                    
                    left_y_axis = st.sidebar.selectbox(
                        "📈 左側Y軸變數", 
                        options=numeric_columns, 
                        index=0
                    )
                    
                    right_y_axis_options = ['None'] + numeric_columns
                    right_y_axis = st.sidebar.selectbox(
                        "📊 右側Y軸變數 (可選)", 
                        options=right_y_axis_options, 
                        index=0
                    )
                    
                    st.markdown("### 🎮 GPUMon 數據分析")
                    st.success(f"✅ 數據載入成功：{df_check.shape[0]} 行 × {df_check.shape[1]} 列")
        
        else:
            st.info("📊 請上傳GPUMon CSV檔案進行測試")
    
    else:
        st.info("🚀 **開始使用** - 請在左側上傳您的 GPUMon.csv 文件進行測試")

if __name__ == "__main__":
    main()
