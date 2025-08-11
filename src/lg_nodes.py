import json
import os
import re
import smtplib
import unicodedata
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import AIMessage, HumanMessage
from langchain_community.vectorstores import Chroma
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.graph import END
from langgraph.types import interrupt

from lg_state import State

load_dotenv()

# ================================= CONFIGURACIÓN =================================

# Variables de entorno
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_DEPLOYMENT_NAME = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
AZURE_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]
MODEL_NAME = os.environ["AZURE_OPENAI_MODEL_NAME"]
CHUNK_SIZE = int(os.environ["CHUNK_SIZE"])

# Configuración para Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
os.environ["OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION

# Inicialización del modelo LLM
llm = init_chat_model(
    MODEL_NAME,
    azure_deployment=AZURE_DEPLOYMENT_NAME,
    model_provider="azure_openai",
    verbose=True, 
)

# Modelo de embeddings
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_key=AZURE_OPENAI_API_KEY
)

# Cargar base de vectores
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# Mapa para convertir numerales romanos a arábigos
_ROMAN_TO_ARABIC = {
    "i":   "1",
    "ii":  "2",
    "iii": "3",
    "iv":  "4",
    "v":   "5",
}

# Cargar datos externos
with open("data/toolCalendarioAcademico/calendarioAcademico2025.json", encoding="utf-8") as f:
    calendar_events = json.load(f)

with open("data/toolExamenesParciales/parcialesOtoño2025.json", encoding="utf-8") as f:
    exams_schedule = json.load(f)

with open("data/toolCombis/combis.json", encoding="utf-8") as f:
    combi_schedules = json.load(f)


# =================================== TOOLS ===================================

@tool(
    name_or_callable="send_email_tool",
    description=(
        "Envía un correo electrónico a alguna casilla oficial de la Universidad de San Andrés "
        "(como alumnos@udesa.edu.ar). Se debe usar cuando el usuario desea contactar con la "
        "universidad por mail. Se debe proporcionar el correo electrónico del estudiante y "
        "la consulta que desea enviar."
    )
)
def send_email_tool(student_email: str, query: str) -> str:
    """
    Envía un mail con la consulta del usuario a una dirección oficial de la universidad.
    """
    # Configuración desde variables de entorno
    email_host = os.environ["EMAIL_HOST"]
    email_port = int(os.environ["EMAIL_PORT"])
    email_username = os.environ["EMAIL_USERNAME"]
    email_password = os.environ["EMAIL_PASSWORD"]
    email_from = os.environ["EMAIL_FROM"]

    email_to = "vigomalusardim@udesa.edu.ar"  # Dirección oficial de la universidad

    # Construcción del mail
    subject = "Consulta enviada desde el Asistente Virtual de UdeSA"
    body = f"El alumno con dirección de correo: {student_email} realizó la siguiente consulta:\n\n{query}"

    message = MIMEMultipart()
    message["From"] = email_from
    message["To"] = email_to
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(email_host, email_port)
        server.starttls()
        server.login(email_username, email_password)
        server.send_message(message)
        server.quit()
        return "Tu consulta fue enviada exitosamente por correo electrónico a la Universidad. Pronto se pondrán en contacto contigo."
    except Exception as e:
        return f"Hubo un error al intentar enviar el correo: {str(e)}"


@tool(
        name_or_callable="condense_query_tool",
        description=(
            "Usar en casos de consultas con referencias a interacciones previas." \
            "Dada una consulta no autonoma que no podria responderse sin contexto," \
            "esta herramienta genera una nueva consulta usando la consulta original y la memoria de la conversación. " \
        )
)
def condense_query_tool(query:str, state: State) -> str:
    """
    Dada una consulta y un contexto, genera una nueva consulta más precisa.
    """
    system_prompt = (
        "Eres un asistente encargado de generar consultas precisas a partir de una consulta original y un contexto"
        "para maximizar la respuesta generada por un modelo de lenguaje."
        "Dado el historial de conversación y la consulta original,"
        "genera una nueva consulta que sea precisa, clara e independiente del contexto."
    )
    user_promtp = (
        f"""Dada la consulta original: "{query}" y el contexto de la conversación,
        genera una nueva consulta que sea precisa y clara, sin depender del contexto.
        El contexto de la conversación es: {(state["messages"])}"""
    )
    full_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_promtp}
    ]
    condensed_query = llm.invoke(full_prompt)
    return condensed_query.content


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]


@tool(
    name_or_callable="retriever_tool",
    description=(
        "Devuelve documentos con información importante de la Universidad de San Andrés "
        "relacionados con las carreras (información general, planes de estudios, "
        "proyección internacional y futuro profesional, etc.)"
    ),
    return_direct=False
)
def retriever_tool(query: str, **kwargs) -> list:
    """
    Recupera documentos relevantes del vectorstore sobre la Universidad de San Andrés.
    """
    custom_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = custom_retriever.invoke(query)
    contenidos = [doc.page_content for doc in docs]
    print("Retrieved documents:", len(contenidos))
    return contenidos


@tool(
    name_or_callable="academic_calendar_tool",
    description=(
        "Para fechas generales del calendario académico de UdeSA: feriados, inicios de semestre, "
        "períodos de parciales, finales, recuperatorios y vacaciones. "
        "Recibe la consulta del usuario y, opcionalmente, la fecha actual en formato 'DD/MM/YYYY'."
    )
)
def academic_calendar_tool(query: str, current_date_str: str = None) -> str:
    """
    Filtra eventos del calendario académico según la consulta del usuario:
    - Por tipo (feriado, plazo, parciales, finales, vacaciones)
    - Por semestre
    - Por rango de fechas o mes
    - Próximo evento
    """
    # Parsear fecha actual si se proporciona
    hoy = None
    if current_date_str:
        try:
            hoy = datetime.strptime(current_date_str, "%d/%m/%Y").date()
        except ValueError:
            pass

    q = query.lower()
    # Detectar tipo de evento si el usuario lo nombra
    tipos = ["feriado", "plazo", "parciales", "finales", "vacaciones", "inicio", "recuperatorios"]
    tipo_buscado = next((t for t in tipos if t in q), None)

    # Filtrar según lo detectado
    resultados = []
    for ev in calendar_events:
        ev_tipo = ev["Tipo"].lower()
        ev_inicio = datetime.strptime(ev["Fecha Inicio"], "%Y-%m-%d %H:%M:%S").date()
        ev_fin = datetime.strptime(ev["Fecha Fin"],   "%Y-%m-%d %H:%M:%S").date()
        ev_sem = ev["Semestre"]
        
        # Filtrar por semestre si se especifica
        if "semestre 1" in q and ev_sem != 1:
            continue
        if "semestre 2" in q and ev_sem != 2:
            continue
        
        # Filtrar por tipo si se especifica
        if tipo_buscado and ev_tipo != tipo_buscado:
            continue
        
        # Si pidió "próximo" y tenemos la fecha de hoy, solo eventos futuros
        if "próxim" in q and hoy and ev_fin < hoy:
            continue
        
        resultados.append((ev["Evento"], ev_inicio, ev_fin, ev_tipo, ev_sem))

    if not resultados:
        return "No encontré eventos que coincidan con tu consulta sobre el calendario académico."

    # Construir la respuesta
    if "próxim" in q and hoy:
        # Si pidió próximo evento, ordenar y tomar el primero
        resultados = sorted(resultados, key=lambda x: x[1])
        ev, ini, fin, _, sem = resultados[0]
        return f"El próximo evento ({ev}) es {ini.strftime('%d/%m/%Y')} (semestre {sem})."
    
    # Listar todos los resultados
    lineas = []
    for ev, ini, fin, tp, sem in resultados:
        if ini == fin:
            fechas = ini.strftime("%d/%m/%Y")
        else:
            fechas = f"{ini.strftime('%d/%m/%Y')} al {fin.strftime('%d/%m/%Y')}"
        lineas.append(f"- {ev} ({tp}) — {fechas}, semestre {sem}")
    
    return "Aquí lo que encontré:\n" + "\n".join(lineas)


def _normalize(text: str) -> str:
    """Quita tildes y convierte a minúsculas."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.lower()


@tool(
    name_or_callable="exam_schedule_and_modality_tool",
    description=(
          "Para fechas, horarios, aulas y modalidad de exámenes de una materia específica: "
          "parciales, finales o recuperatorios. Ejemplo: “parcial de Física I”, “final de Base de Datos”, “examen de P042”."
    )
)
def exam_schedule_and_modality_tool(query: str, current_date_str: Optional[str] = None) -> str:
    """
    Busca información de exámenes según la consulta del usuario.
    """
    # Normalizar query
    q = _normalize(query)
    
    # Parsear año si lo menciona
    year_filter = None
    for tok in q.split():
        if tok.isdigit() and len(tok) == 4:
            year_filter = int(tok)
            break

    # Clasificar eventos
    date_exams = []   # (fecha, ev)
    domic_exams = []  # ev
    no_exams = []     # ev

    for ev in exams_schedule:
        # Filtrar por año si se solicitó
        if year_filter and ev.get("Año") != year_filter:
            continue

        raw = ev.get("Fecha")
        norm_raw = _normalize(str(raw or ""))

        if norm_raw in ("sin examen", "", "nan"):
            no_exams.append(ev)
            continue
        if "domiciliari" in norm_raw:
            domic_exams.append(ev)
            continue
        
        # Fecha concreta
        try:
            fecha = datetime.strptime(raw.split()[0], "%Y-%m-%d").date()
            date_exams.append((fecha, ev))
        except:
            continue

    # 1) Pregunta por modalidad
    if "modalidad" in q or "domiciliari" in q or "sin examen" in q:
        líneas = []
        if "domiciliari" in q:
            if not domic_exams:
                return "No se encontraron exámenes domiciliarios."
            for ev in domic_exams:
                líneas.append(f"- {ev['Codigo Materia']} ({ev['Materia']}) — domiciliario")
            return "Exámenes domiciliarios:\n" + "\n".join(líneas)
        
        if "sin examen" in q:
            if not no_exams:
                return "Todas las materias tienen examen programado o modalidad domiciliaria."
            for ev in no_exams:
                líneas.append(f"- {ev['Codigo Materia']} ({ev['Materia']}) — sin examen")
            return "Materias sin examen:\n" + "\n".join(líneas)

    # 2) Pregunta por parciales / finales / recuperatorios
    if any(k in q for k in ("parcial", "final", "recuperatori")):
        tipo = "parciales" if "parcial" in q else "finales" if "final" in q else "recuperatorios"
        if not date_exams:
            return f"No hay fechas de {tipo} registradas."
        
        resultados = sorted(date_exams, key=lambda x: x[0])
        líneas = [
            f"- {ev['Codigo Materia']} ({ev['Materia']}): {fecha.strftime('%d/%m/%Y')} a las {ev.get('Hora','?')} en {ev.get('Aulas','?')}"
            for fecha, ev in resultados
        ]
        return f"Fechas de {tipo}:\n" + "\n".join(líneas)

    # 3) Match por código o nombre completo, elegir el candidato con nombre más largo
    candidates = []  # (longitud_nombre, fecha, ev)
    for fecha, ev in date_exams:
        code = _normalize(ev.get("Codigo Materia", ""))
        name = _normalize(ev.get("Materia", ""))
        
        # Armar versión arábiga si hay sufijo romano
        parts = name.split()
        alt_name = None
        if parts and parts[-1] in _ROMAN_TO_ARABIC:
            alt_name = " ".join(parts[:-1] + [_ROMAN_TO_ARABIC[parts[-1]]])

        # Patrón exacto de palabra
        pat_name = rf"\b{re.escape(name)}\b"
        pat_alt  = rf"\b{re.escape(alt_name)}\b" if alt_name else None

        # 1) Código exacto
        if code == q:
            candidates.append((len(code), fecha, ev))
            continue
        
        # 2) Nombre completo
        if re.search(pat_name, q):
            candidates.append((len(name), fecha, ev))
            continue
        
        # 3) Versión numérica "Física 2"
        if pat_alt and re.search(pat_alt, q):
            candidates.append((len(alt_name), fecha, ev))

    if candidates:
        # Elegir el match con nombre más largo
        _, fecha, ev = max(candidates, key=lambda x: x[0])
        modalidad = (
            "domiciliario"
            if "domiciliari" in _normalize(str(ev.get("Fecha", "")))
            else "presencial"
        )
        return (
            f"Examen de {ev['Codigo Materia']} — {ev['Materia']}:\n"
            f"- Fecha: {fecha.strftime('%d/%m/%Y')}\n"
            f"- Horario: {ev.get('Hora','?')}\n"
            f"- Aula: {ev.get('Aulas','?')}\n"
            f"- Modalidad: {modalidad}"
        )

    # 4) Próximo examen general
    if "próxim" in q and date_exams:
        hoy = None
        if current_date_str:
            try:
                hoy = datetime.strptime(current_date_str, "%d/%m/%Y").date()
            except:
                pass
        
        futuros = [(f, ev) for f, ev in date_exams if not hoy or f >= hoy]
        if futuros:
            f, ev = min(futuros, key=lambda x: x[0])
            return (
                f"Próximo examen: {ev['Codigo Materia']} – {ev['Materia']} el "
                f"{f.strftime('%d/%m/%Y')} a las {ev.get('Hora','?')} en {ev.get('Aulas','?')}."
            )
        return "No hay más exámenes con fecha futura."

    # 5) Si no matcheó
    return (
        "Perdón, no entendí. Podés pedirme:\n"
        "- '¿Cuándo es el examen de Física I?'\n"
        "- '¿Qué modalidad tuvo el examen de Física I en 2024?'\n"
        "- '¿En qué horario se rinde Base de Datos y en qué aula?'\n"
        "- '¿Cuál es el próximo examen?'\n"
        "- 'Exámenes domiciliarios' / 'Materias sin examen'\n"
    )


@tool(
    name_or_callable="combi_schedule_tool",
    description=(
        "Para preguntas sobre los horarios de la combi que sale desde 'Estación Victoria' "
        "o 'Campus UdeSA'. Devuelve las horas de salida según la parada solicitada. "
        "Debe usarse con la hora actual para encontrar el próximo horario disponible."
    )
)
def combi_schedule_tool(query: str, current_time_str: str = None) -> str:
    """
    Dada una consulta del usuario y la hora actual, detecta en qué parada le interesa el horario
    y devuelve la lista de horas de salida correspondientes, incluyendo la combi más cercana en tiempo.
    `current_time_str` debe estar en formato "HH:MM".
    """
    q = query.lower()
    current_time = None
    
    if current_time_str:
        try:
            # Parsear current_time_str a un objeto time
            current_time = datetime.strptime(current_time_str, "%H:%M").time()
        except ValueError:
            pass  # Si falla el parseo, continuar sin hora actual

    # Buscar cuál de las paradas menciona el usuario
    for parada, horarios in combi_schedules.items():
        if parada.lower() in q:
            response_lines = []
            
            if current_time:
                # Encontrar la próxima combi más cercana
                next_combi_time = None
                for h_str in horarios:
                    try:
                        h_time = datetime.strptime(h_str, "%H:%M").time()
                        if h_time > current_time:
                            if next_combi_time is None or h_time < next_combi_time:
                                next_combi_time = h_time
                    except ValueError:
                        continue  # Saltar strings de tiempo mal formados

                if next_combi_time:
                    response_lines.append(f"La próxima combi desde *{parada}* sale a las {next_combi_time.strftime('%H:%M')}.")
                else:
                    response_lines.append(f"No hay más combis programadas para hoy desde *{parada}* después de tu hora actual.")
                
                response_lines.append("Aquí están todos los horarios para hoy:")
            else:
                response_lines.append(f"Horarios de salida de la combi desde *{parada}*:")

            lista_horas = "\n".join(f"- {h}" for h in horarios)
            response_lines.append(lista_horas)
            return "\n".join(response_lines)

    # Si no detectó ninguna parada, dar pista de uso
    return (
        "No entendí desde qué parada quieres el horario. "
        "Por favor menciona 'Estación Victoria' o 'Campus UdeSA' en tu pregunta."
    )


# Lista de todas las herramientas
tools = [
    human_assistance,
    retriever_tool,
    combi_schedule_tool,
    condense_query_tool,
    send_email_tool,
    academic_calendar_tool,
    exam_schedule_and_modality_tool
]

llm_with_tools = llm.bind_tools(tools)


# =================================== NODOS ===================================

def planner_node(state: dict) -> dict:
    """
    Nodo que planifica paso a paso cómo resolver una pregunta compleja.
    Devuelve un mensaje con los pasos pensados antes de responder.
    """
    print("Running planner_node...")
    
    user_message = state["messages"][-1]
    if not user_message or not isinstance(user_message, HumanMessage):
        return {"messages": []}

    tools_description = """
        Tenés disponibles las siguientes herramientas que puede usar el modelo:

        - `retriever_tool`: Para recuperar información documental de la universidad (UdeSA/Universidad de San Andrés), como materias, carreras, reglamentos, info de intercambios, etc.
        - `combi_schedule_tool`: Para ver los horarios de la combi desde 'Estación Victoria' o 'Campus UdeSA'. Necesita que le pases la hora actual como argumento `current_time_str`.
        - `academic_calendar_tool`: Para responder sobre fechas importantes como feriados, parciales o vacaciones. Requiere que le pases `current_date_str`.
        - `exam_schedule_and_modality_tool`: Para ver fechas, horarios, aulas o modalidad de exámenes. También necesita `current_date_str`.
        - `send_email_tool`: Para redactar y enviar un correo. Necesita la dirección de correo del alumno y el contenido del correo. NO se debe buscar en el rag el correo destino.

        Siempre que puedas, indicá cuál tool se debe usar en cada paso del plan, y con qué parámetros.
    """

    query = user_message.content

    planner_prompt = (
        "Sos un asistente inteligente de la Universidad de San Andrés. "
        "Tu tarea es planificar cómo responder preguntas complejas. "
        "Dado un mensaje del usuario, descomponelo en pasos claros e indica qué herramienta debe usarse en cada paso, si corresponde. "
        "No respondas la consulta, solo generá un plan. "
        "El objetivo es que otro modelo pueda seguir el plan paso a paso y responder correctamente.\n\n"
        f"{tools_description}\n\n"
        f"Consulta del usuario: {query}\n\n"
        "Plan de acción paso a paso:"
    )

    plan = llm.invoke([{"role": "system", "content": planner_prompt}])

    # Crear un nuevo mensaje humano que diga: "Resuelve mi consulta anterior siguiendo este plan: ..."
    followup_msg = HumanMessage(
        content=f"Resuelve mi consulta anterior siguiendo este plan:\n\n{plan.content}"
    )

    # Devolver el nuevo estado con el mensaje original reemplazado por el nuevo HumanMessage
    return { "messages": [followup_msg] }


def chatbot(state: dict) -> dict:
    """
    El nodo principal del chatbot que maneja consultas de usuarios y gestiona el estado de la conversación.
    Usa el LLM con herramientas para responder a consultas de usuarios basándose en el historial de conversación.
    """
    print("Running chatbot...")
    print("\n\nState messages que recibe el chatbot:", state["messages"])

    now = datetime.now()
    current_time_str = now.strftime("%H:%M")
    current_date_str = now.strftime("%d/%m/%Y")

    system_prompt = (
    "Sos un Asistente Virtual experto de la Universidad de San Andrés (UdeSA) "
    f"La fecha actual es: {current_date_str} y la hora actual es: {current_time_str}. "
    "Estás diseñado para ayudar a responder preguntas sobre carreras, materias, inscripciones, horarios, trámites y demás información relevante de la universidad. "
    "Podés usar herramientas externas para obtener información más precisa o resolver tareas complejas. "
    "Tu objetivo es resolver de forma efectiva las consultas del usuario, usando las herramientas solo cuando sea necesario. "

    "\n\n### Comportamiento esperado:\n"
    "- Respondé de forma clara, precisa y concisa.\n"
    "- No adivines ni inventes datos. Si no sabés algo, usá una herramienta o indicá que no se encontró la información.\n"
    "- Si la consulta se refiere a algo que no está en el mensaje actual pero puede estar en mensajes anteriores, usá `condense_query_tool` para generar una consulta clara.\n"
    "- Si la pregunta requiere datos documentales de la universidad de San Andres (UdeSA), usá `retriever_tool`.\n"
    "- Si la pregunta es sobre horarios de las combis, usá `combi_schedule_tool`. CUANDO USES combi_schedule_tool, DEBES PASARLE EL ARGUMENTO 'current_time_str' CON LA HORA ACTUAL EN FORMATO HH:MM.\n" \
    "- Si la pregunta es sobre el calendario académico (feriados, parciales, vacaciones, plazos), usá `academic_calendar_tool` y pásale 'current_date_str' con la fecha actual en formato DD/MM/YYYY.\n"
    "- Si la pregunta es sobre fechas, horario, aula o modalidad de un examen de materia específica usá ` exam_schedule_and_modality_tool` y pasale 'current_date_str' con la fecha actual en formato DD/MM/YYYY.\n"

    "\n### Formato de respuesta:\n"
    "- Podés responder directamente si tenés la información suficiente.\n"
    "- Si necesitás ayuda de una herramienta, hacé una llamada a la tool correspondiente.\n"

    "\n### Ejemplos de routing de tools:\n"
    "- “¿Cuándo es el periodo de parciales?”           → academic_calendar_tool"
    "- “¿Qué feriados hay este mes?”                    → academic_calendar_tool"
    "- “¿Cuándo empiezan las vacaciones de invierno?”  → academic_calendar_tool"


    "- “¿Cuándo es el parcial de Física I?”            → exam_schedule_and_modality_tool"
    "- “¿En qué aula rindo Base de Datos?”             → exam_schedule_and_modality_tool"
    "- “¿Qué modalidad tuvo el examen de P042 en 2024?”→ exam_schedule_and_modality_tool"

    "- Recordá siempre ser cordial en tus respuestas. No uses lenguaje ofensivo ni despectivo. "
        "- Usá vocabulario argentino y rioplatense, no uses vocabulario español de España ni de países centroamericanos. "
    )

    # Combinar el prompt del sistema con todo el historial de mensajes del estado
    full_prompt = [
        {"role": "system", "content": system_prompt},
        *state["messages"]
    ]

    # Invoke the LLM with the complete prompt
    message = llm_with_tools.invoke(full_prompt)

    print("LLM response:", message)
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    print("Cantidad de tool calls:", len(message.tool_calls))
    print("Tool calls in message:", message.tool_calls)
    
    return {"messages": [message]}


class BasicToolNode:
    """Un nodo que ejecuta las herramientas solicitadas en el último AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]  # Obtener el último mensaje de la lista
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            ) # Call the tool with the arguments provided in the tool call
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


# ========================== CONDITIONAL EDGES ==========================

def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode si the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list): # If state is a list, get the last message
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# ========================== HELPER FUNCTIONS ==========================

def is_complex_question(query: str) -> bool:
    """
    Determina si una consulta es compleja y requiere múltiples pasos para ser respondida.
    """
    prompt = f"""
        Sos un clasificador de consultas. Dada la siguiente consulta, respondé únicamente con "sí" si la consulta es compleja (es decir, contiene múltiples subconsultas o requiere múltiples pasos para ser respondida), o "no" si es simple y se puede resolver con una sola herramienta.

        Consulta: {query}

        ¿Es compleja?
    """
    result = llm.invoke(prompt)
    print("is_complex_question result:", result.content)
    return ("sí" in result.content.lower() or "si" in result.content.lower())
