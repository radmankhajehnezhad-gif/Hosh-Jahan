import os
import gradio as gr
from transformers import pipeline

# Lightweight default model for Replit. Change MODEL_ID if you want another model.
MODEL_ID = os.environ.get("MODEL_ID", "google/flan-t5-base")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Use text2text-generation pipeline for instruction-following models like FLAN
generator = pipeline("text2text-generation", model=MODEL_ID, use_auth_token=HF_TOKEN)

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

def chat_with_model(history, user_message):
    history = history or []
    # Build a concise prompt
    prompt = SYSTEM_PROMPT + "\n"
    for u, a in history:
        prompt += f"User: {u}\nAssistant: {a}\n"
    prompt += f"User: {user_message}\nAssistant:"

    out = generator(prompt, max_length=256, do_sample=True, top_p=0.95, temperature=0.7)
    reply = out[0].get("generated_text", "")
    history.append((user_message, reply))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Dark Room — Public AI Chat (Replit / Render friendly)\n\nType a message and press Enter.")
    chat = gr.Chatbot()
    state = gr.State([])
    txt = gr.Textbox(show_label=False, placeholder="پیامت رو بنویس و Enter بزن")
    txt.submit(chat_with_model, [state, txt], [chat, state])

if __name__ == "__main__":
    # Bind to all interfaces and use PORT env var (Render/Replit friendly)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))