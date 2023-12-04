import streamlit as st
import base64
import json
import nest_asyncio
from llama_index import (LLMPredictor, ServiceContext, StorageContext,
                         load_index_from_storage, set_global_service_context)
from llama_index.llms import LlamaCPP
from llama_index.memory import ChatMemoryBuffer

nest_asyncio.apply()

# Configuración de la página
st.set_page_config(page_title="UTPL Chat", page_icon="🤖", layout="wide")

# Función para cargar imágenes como base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Cargar imagen
img = get_img_as_base64("utpl.jpeg")

# Estilo de la aplicación
st.markdown("""
    <style>
        .stApp {
            background-color: black;
            background-attachment: fixed;
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title("UTPL Chat 🤖")

# Configuración del modelo Llama
llm = LlamaCPP(
    model_path="./models/llama-2-13b-chat.Q4_0.gguf", 
    temperature=0.0, 
    model_kwargs={'n_gpu_layers': 10},
    max_new_tokens=500
)

llm_predictor = LLMPredictor(llm=llm)

# Contexto de almacenamiento y servicio
storage_context = StorageContext.from_defaults(persist_dir="./storage")
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
set_global_service_context(service_context=service_context)

# Cargar índice desde el almacenamiento
index = load_index_from_storage(storage_context, service_context=service_context)

# Inicializar la memoria del chat
if 'memory' not in st.session_state:
    st.session_state['memory'] = ChatMemoryBuffer.from_defaults(token_limit=4096)

# Crear motor de chat
chat_engine = index.as_chat_engine(
    chat_mode='context', 
    simularity_top_k=5,
    memory=st.session_state['memory'],
    system_prompt="Responde amablemente la pregunta en base al contexto la data sobre las carreras de la Universidad Técnica Particular de Loja (UTPL)",
)

# Inicializar mensajes
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hola! ¿Qué quieres saber de la UTPL?"}]

# Mostrar mensajes anteriores
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Obtener la pregunta del usuario
if prompt := st.text_input("¿Qué quieres saber sobre la UTPL?"):
    st.session_state['messages'].append({"role": "user", "content": prompt})

    with st.chat_message("user"): 
        st.markdown(prompt)

    is_async = True
    
    with st.chat_message("assistant"): 
        with st.spinner("Espere un momento..."):
            if is_async:
                # Obtener respuesta del motor de chat de forma asíncrona
                response = chat_engine.stream_chat(prompt)
                placeholder = st.empty()
                
                full_response = ''
                for item in response.response_gen:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

            else:
                # Obtener respuesta del motor de chat de forma síncrona
                full_response = chat_engine.chat(prompt)
                st.markdown(full_response)

        # Agregar la respuesta al historial de mensajes
        st.session_state['messages'].append({"role": "assistant", "content": full_response})
