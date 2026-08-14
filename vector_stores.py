from langchain_chroma import Chroma
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings

class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=embedding,
            persist_directory=config.persist_directory
        )

    def get_retriever(self):
        # '''返回向量检索器方便加入chain'''
        return self.vector_store.as_retriever(search_kwargs={'k': config.similarity_threshold})

if __name__ == "__main__":
    retriever = VectorStoreService(embedding=DashScopeEmbeddings(model='text-embedding-v1')).get_retriever()
    res = retriever.invoke('我的体重是140斤,推荐尺码')
    print(res)
