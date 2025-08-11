import os
import requests
from flask import Flask, request
from dotenv import load_dotenv

from lg_agent import stream_graph_updates

import os

# PRUEBA: fuerza un token “limpio” directamente
os.environ['TELEGRAM_TOKEN'] = '7652470082:AAEffFpBHV9NN8SOTy6FVBm3NyGXOsAKSO0'

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{TOKEN}"

app = Flask(__name__)

def mi_chatbot_responder(texto, chat_id):

    #return f"Eco: {texto}"
    return stream_graph_updates(texto, chat_id)

@app.route('/webhook/telegram', methods=['POST'])
@app.route('/webhook/telegram', methods=['POST'])
def telegram_webhook():
    data = request.get_json(force=True)
    print("📨 Update recibido:", data)

    mensaje = data.get('message')
    if not mensaje or 'text' not in mensaje:
        print("⚠️ No era un mensaje de texto válido.")
        return '', 200

    chat_id = mensaje['chat']['id']
    texto   = mensaje['text']
    respuesta = mi_chatbot_responder(texto, chat_id)

    payload = {'chat_id': chat_id, 'text': respuesta}
    print("⏩ Enviando a Telegram:", payload)

    try:
        url = f"{BASE_URL}/sendMessage"
        print("▶️ repr(BASE_URL) =", repr(BASE_URL))
        print("▶️ repr(url)      =", repr(url))
        print("▶️ url             =", url)
        r = requests.post(url, json=payload)
        print("⬅️ status_code, text:", r.status_code, r.text)



    except Exception as e:
        print("❌ Error al llamar a sendMessage:", e)

    return '', 200


if __name__ == '__main__':
    # Ejecuta en el puerto 5000 de tu localhost
    app.run(port=5000)
