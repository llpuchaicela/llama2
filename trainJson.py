from llama_index import VectorStoreIndex, SimpleDirectoryReader, LLMPredictor, set_global_service_context, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.query_engines import JSONQueryEngine
from IPython.display import Markdown, display

# Inicializar el modelo LlamaCPP
llm = LlamaCPP(model_path="./models/llama-2-13b-chat.Q4_0.gguf")
llm_predictor = LLMPredictor(llm=llm)

# Configurar el contexto de servicio global
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
set_global_service_context(service_context=service_context)

# Cargar documentos
documents = SimpleDirectoryReader('data').load_data()

# Crear un índice a partir de los documentos
index = VectorStoreIndex.from_documents(documents, show_progress=True)

# Crear un motor de consulta
query_engine = index.as_query_engine()

# Consulta de NL
nl_response = query_engine.query("De qué trata la carrera de turismo")
print(nl_response)

# Consulta de JSON sin síntesis de respuesta
raw_response = query_engine.query("Qué materias tiene el primer ciclo de la carrera de computación?", synthesize_response=False)
print(raw_response)

# Mostrar respuestas
display(Markdown(f"<h1>Natural language Response</h1><br><b>{nl_response}</b>"))
display(Markdown(f"<h1>Raw JSON Response</h1><br><b>{raw_response}</b>"))

# Persistir el índice
index.storage_context.persist("./storage")
