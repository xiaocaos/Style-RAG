md5_path = 'md5.text'

# chroma
collection_name = 'RAG'
persist_directory = './chroma-db'

chunk_size = 1000
chunk_overlap = 100
separators = ['\n\n','\n','.','?','!','。','？','！',' ','']

max_split_str_number = 1000


similarity_threshold = 1

embedding_chat_model = 'text-embedding-v1'
chat_model = 'qwen-turbo'

session_config = {
        'configurable':{'session_id':'user_001'},
    }