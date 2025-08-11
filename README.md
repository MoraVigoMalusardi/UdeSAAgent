# TPFinal\_NLP

**Repositorio:** [https://github.com/anacotler/TPFinal\_NLP.git](https://github.com/anacotler/TPFinal_NLP.git)

## Descripción

Este proyecto presenta un agente conversacional construido con LangGraph y LangChain para asistir a estudiantes de la Universidad de San Andrés en consultas relacionadas con su vida académica. Combina recuperación de información, planificación multi-step y herramientas especializadas para responder consultas reales.

## Tabla de contenidos

* [Requisitos](#requisitos)
* [Instalación](#instalación)
* [Configuración](#configuración)
* [Generar Base de Datos](#generar-base-de-datos)
* [Uso](#uso)

  * [Agente Conversacional (CLI)](#agente-conversacional-cli)
* [Estructura de Carpetas](#estructura-de-carpetas)
* [Contribuir](#contribuir)
* [Licencia](#licencia)

## Requisitos

* Python 3.10+
* Virtual environment (venv)
* Variables de entorno definidas en `.env`
* ChromaDB local

## Instalación

```bash
git clone https://github.com/anacotler/TPFinal_NLP.git
cd TPFinal_NLP
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

## Configuración

Copia el template de entorno y complétalo:

```bash
cp .env.example .env
```

Edita `.env` y reemplaza cada placeholder con tus credenciales y rutas reales.

## Generar Base de Datos

Antes de ejecutar el agente, genera la base de datos en ChromaDB:

```bash
python scripts/generar_base_de_datos.py
```

## Uso

### Agente Conversacional (CLI)

```bash
python src/lg_agent.py
```

* Ingresa tus preguntas en el prompt `Pregunta>`.
* Para terminar la conversación, escribe `quit` y presiona Enter.

## Estructura de Carpetas

```
TPFINAL_NLP/
├── chroma_db/                   # Base de datos de ChromaDB (persistente)
├── data/                        # Documentos del agente
│   └── agent_datasets/          # JSONs que usa el agente en tiempo de ejecución
├── evaluations/                 # Evaluaciones por herramienta
│   └── retriever/
│       ├── data/                # Datasets para tests del retriever
│       │   └── eval_dataset.json
│       └── results/             # Resultados de evaluación (CSV, gráficos, etc.)
├── scripts/                     # Utilidades y scripts de mantenimiento
│   └── generar_base_de_datos.py # Script para poblar ChromaDB
├── src/                         # Código fuente
│   ├── app.py                   # Flask-app (servidor HTTP)
│   ├── lg_agent.py              # Entry-point CLI del agente conversacional
│   ├── lg_nodes.py              # Definición de nodos de LangGraphAgent
│   └── lg_state.py              # Estado y lógica interna de LangGraphAgent
├── venv/                        # Entorno virtual (no versionar)
├── .env                         # Variables de entorno (local)
├── .env.example                 # Template de variables de entorno
├── README.md                    # Documentación (este archivo)
└── requirements.txt             # Dependencias pip
```


