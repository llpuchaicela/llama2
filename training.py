from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, set_global_service_context, ServiceContext
from llama_index.llms import LlamaCPP

llm = LlamaCPP(model_path="./models/llama-2-13b-chat.Q4_0.gguf")
llm_predictor = LLMPredictor(llm=llm)

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

set_global_service_context(service_context=service_context)

documents = SimpleDirectoryReader('data').load_data()

index = VectorStoreIndex.from_documents(documents, show_progress=True)

query_engine = index.as_query_engine()
 
response = query_engine.query("¿Qué tal es la carrera de Administración de Empresas?")
print(response)
response = query_engine.query("¿Qué tal es la carrera de Comercio Exterior?")
print(response)

index.storage_context.persist("./storage")
