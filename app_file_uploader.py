import streamlit as st
from knowledge_base import KnowledgeBaseService
import time
st.title('知识库更新服务')
updater_file = st.file_uploader(
    label='请上传txt文件',
    type=['txt'],
    accept_multiple_files=False,
)

service = KnowledgeBaseService()
if 'service' not in st.session_state:
    st.session_state['service'] = KnowledgeBaseService()

if updater_file is not None:
    file_name = updater_file.name
    file_type = updater_file.type
    file_size = updater_file.size /1024

    st.subheader(f'文件名:{file_name}')
    st.write(f'格式:{file_type} | 大小:{file_size:.2f}KB')

    text = updater_file.getvalue().decode('utf-8')

    with st.spinner('文件载入数据库中...'):
        time.sleep(2)
        result = st.session_state['service'].update_by_str(text,file_name)
        st.write(result)