import streamlit as st
import sys,os

# 把当前脚本所在的目录加入 Python 模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag import RagService
import config_data as config


st.title('AI小爽客服')
st.divider()#分隔符

#在页面下方提供用户输入框
prompt = st.chat_input()

if 'message' not in st.session_state:
    st.session_state['message']= [{'role':'assistant','content':'我是AI助手小爽，请问有什么可以帮助你的吗?'}]

if 'rag' not in st.session_state:
    st.session_state['rag'] = RagService()

for message in st.session_state['message']:
    st.chat_message(message['role']).write(message['content'])

if prompt:
    st.chat_message('user').write(prompt)#在页面输出用户提问
    st.session_state['message'].append({'role':'user','content':prompt})

    with st.spinner('小爽正在努力思考：'):
        res_stream = st.session_state['rag'].chain.stream({'input':prompt},config.session_config)
        res = st.chat_message('assistant').write_stream(res_stream)
        st.session_state['message'].append({'role':'assistant','content':res})


