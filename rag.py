from langchain_core.runnables import RunnablePassthrough,RunnableWithMessageHistory,RunnableLambda
from vector_stores import VectorStoreService
import config_data as config
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
import os

from file_history_store import get_history_chat


str_parser = StrOutputParser()
def print_prompt(prompt):
    print('='*25)
    print(prompt.to_string())
    print('='*25)
    return prompt

def format_for_retriever(value: dict) -> str:
    return value["input"]


def format_for_prompt_template(value):
    # {input, context, history}
    new_value = {}
    new_value["input"] = value["input"]["input"]
    new_value["context"] = value["context"]
    new_value["history"] = value["input"]["history"]
    return new_value

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(embedding = DashScopeEmbeddings(model=config.embedding_chat_model))
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ('system','以我提供的资料{context}为主，简洁专业的回答用户的问题。'),
                ('system','并且我提供了用户的历史聊天记录如下:'),
                MessagesPlaceholder('history'),
                ('user','请回答用户提问{input}')
            ]
        )
        self.chat_model = ChatOpenAI(
            model=config.chat_model,
            streaming=True,
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()
        def format_document(docs):
            if not docs:
                return "无相关参考资料"
            else:
                format_str = ''
                for chunk in docs:
                    format_str += f"文档片段:{chunk.page_content}\n文档元数据{chunk.metadata}\n\n"
                return format_str

        chain = (
            {'input':RunnablePassthrough(),'context':RunnableLambda(format_for_retriever) | retriever | format_document }
            | RunnableLambda(format_for_prompt_template)
            | self.prompt_template
            | print_prompt
            | self.chat_model
            | str_parser

        )
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history_chat,
            input_messages_key='input',
            history_messages_key='history'
        )

        return conversation_chain

if __name__ == '__main__':
    session_config = {
        'configurable':{'session_id':'user_001'},
    }
    #增强链的invoke为字典
    res = RagService().chain.invoke({'input':'我身高180cm,尺码推荐'},session_config)
    print(res)
