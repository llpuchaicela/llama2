import base64
import nest_asyncio
import streamlit as st
from llama_index import (LLMPredictor, ServiceContext, StorageContext,
                         load_index_from_storage, set_global_service_context)
from llama_index.llms import LlamaCPP
from llama_index.memory import ChatMemoryBuffer

nest_asyncio.apply()


# https://scontent.fuio13-1.fna.fbcdn.net/v/t1.6435-9/120996839_10158988589315921_8361322757228136300_n.jpg?_nc_cat=102&ccb=1-7&_nc_sid=e3f864&_nc_eui2=AeHUhll5E2tIEGzo7Jg--N8eBIvqh7KHjPwEi-qHsoeM_KW47QBC1Y_5XRVf1xtDOhB6KzK5neZo5z5VTNLJZGFO&_nc_ohc=HZNG6BSyHqcAX8Q68Xm&_nc_ht=scontent.fuio13-1.fna&oh=00_AfDmXURCX1gV_1sK893NnvfD4T_LfXpZlDMa3xqSWZziVw&oe=653279B1
st.set_page_config(page_title="UTPL Chat", page_icon="🤖", layout="wide", )   
  

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("utpl.jpeg")

st.markdown(f"""
            <style>
            .stApp {{
                     background-color: black;
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)


st.title("UTPL Chat 🤖")


# /Users/becario/Library/Caches/llama_index/models/llama-2-70b-chat.ggmlv3.q4_K_S.bin
llm = LlamaCPP(
    model_path="./models/llama-2-13b-chat.Q4_0.gguf", 
    temperature=0, 
    model_kwargs={'n_gpu_layers': 10},
    max_new_tokens=500,
    # creativity=0  # Ajusta este valor según tus preferencias
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
    # system_prompt="Responde amablemente la pregunta en base al contexto la data sobre las carreras de la Universidad Técnica Particular de Loja (UTPL)",

system_prompt = (
    "Por favor, responde amablemente a la pregunta considerando el contexto y la información disponible sobre las carreras de la Universidad Técnica Particular de Loja (UTPL). "
    "Puedes proporcionar detalles sobre programas académicos, números de ciclos, descripciones de carreras, perfiles profesionales u otra información relevante que esté exclusivamente en el archivo de datos. "
    "Si no tienes información sobre el tema, simplemente indícalo en tu respuesta."
),
)
# similarity_top_k=1
# Inicializar mensajes
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "Hola! Qué quieres saber de la UTPL?"}]

# Mostrar mensajes anteriores
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
# Obtener la pregunta del usuario
if prompt := st.chat_input("¿Qué quieres saber sobre la UTPL?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

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
        st.session_state.messages.append({"role": "assistant", "content": full_response})


