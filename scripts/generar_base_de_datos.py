import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Carga de variables de entorno
def load_config():
    load_dotenv()
    config = {
        "AZURE_OPENAI_API_KEY": os.environ["AZURE_OPENAI_API_KEY"],
        "AZURE_OPENAI_ENDPOINT": os.environ["AZURE_OPENAI_ENDPOINT"],
        "AZURE_OPENAI_API_VERSION": os.environ["AZURE_OPENAI_API_VERSION"],
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        "CHROMA_DB_DIR": os.environ["CHROMA_DB_DIR"],
        # Directorios por carrera
        "CAREER_DIRS": {
            "abogacia": os.environ.get("DATA_DIR_ABOGACIA"),
            "comunicacion": os.environ.get("DATA_DIR_COMUNICACION"),
            "diseno": os.environ.get("DATA_DIR_DISENO"),
            "economia": os.environ.get("DATA_DIR_ECONOMIA"),
            "economia_empresarial": os.environ.get("DATA_DIR_ECONOMIA_EMPRESARIAL"),
            "finanzas": os.environ.get("DATA_DIR_FINANZAS"),
            "humanidades": os.environ.get("DATA_DIR_HUMANIDADES"),
            "ingenieria_en_biotecnologia": os.environ.get("DATA_DIR_INGENIERIA_EN_BIOTECNOLOGIA"),
            "ingenieria_en_inteligencia_artificial": os.environ.get("DATA_DIR_INGENIERIA_EN_INTELIGENCIA_ARTIFICIAL"),
            "ingenieria_en_sustentabilidad": os.environ.get("DATA_DIR_INGENIERIA_EN_SUSTENTABILIDAD"),
            "administracion": os.environ.get("DATA_DIR_ADMINISTRACION"),
            "ciencia_politica_y_gobierno": os.environ.get("DATA_DIR_CIENCIA_POLITICA_Y_GOBIERNO"),
            "ciencias_de_la_educacion": os.environ.get("DATA_DIR_CIENCIAS_DE_LA_EDUCACION"),
            "ciencias_del_comportamiento": os.environ.get("DATA_DIR_CIENCIAS_DEL_COMPORTAMIENTO"),
            "negocios_digitales": os.environ.get("DATA_DIR_NEGOCIOS_DIGITALES"),
            "relaciones_internacionales": os.environ.get("DATA_DIR_RELACIONES_INTERNACIONALES"),
            "profesorado_educacion_primaria": os.environ.get("DATA_DIR_PROFESORADO_EDUCACION_PRIMARIA"),
            "programas_internacionales": os.environ.get("DATA_DIR_PROGRAMAS_INTERNACIONALES"),
            "programas_internacionales_estudiantes_extranjeros": os.environ.get("DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_EXTRANJEROS"),
            "programas_internacionales_estudiantes_locales": os.environ.get("DATA_DIR_PROGRAMAS_INTERNACIONALES_PARA_ESTUDIANTES_LOCALES"),
            "info_carreras_de_grado": os.environ.get("DATA_DIR_INFO_CARRERAS_DE_GRADO"),
            "catedra_eeuu": os.environ.get("DATA_DIR_CATEDRA_EEUU"),
            "becas_y_asistencia_financiera": os.environ.get("DATA_DIR_BECAS_Y_ASISTENCIA_FINANCIERA"),
            "desarrollo_profesional": os.environ.get("DATA_DIR_DESARROLLO_PROFESIONAL"),
        }
    }
    return config

# Crea el modelo de embeddings
def get_embedding_model(config):
    return AzureOpenAIEmbeddings(
        azure_deployment   = config["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        openai_api_version = config["AZURE_OPENAI_API_VERSION"],
        azure_endpoint     = config["AZURE_OPENAI_ENDPOINT"],
        openai_api_key     = config["AZURE_OPENAI_API_KEY"],
    )

# Carga .txt y convierte en Document con metadata
def cargar_txt_como_documentos(directorio_txt, carrera):
    documentos = []
    if not directorio_txt or not os.path.isdir(directorio_txt):
        return documentos
    for fichero in os.listdir(directorio_txt):
        if fichero.lower().endswith(".txt"):
            path = os.path.join(directorio_txt, fichero)
            loader = TextLoader(path, encoding="utf-8")
            partes = loader.load()
            texto = "\n".join([p.page_content for p in partes])
            documentos.append(Document(
                page_content=texto,
                metadata={"source": fichero, "carrera": carrera}
            ))
    return documentos

# Construye un índice unificado con todos los documentos
def build_unified_index():
    config = load_config()
    embedding_model = get_embedding_model(config)
    all_docs = []
    for carrera, carpeta in config["CAREER_DIRS"].items():
        docs = cargar_txt_como_documentos(carpeta, carrera)
        all_docs.extend(docs)
    print(f"[i] Documentos totales: {len(all_docs)}")
    vectordb = Chroma.from_documents(
        documents         = all_docs,
        embedding         = embedding_model,
        persist_directory = config["CHROMA_DB_DIR"]
    )
    vectordb.persist()
    print(f"[✔] Base de datos Chroma guardada en: {config['CHROMA_DB_DIR']}")

if __name__ == "__main__":
    build_unified_index()
