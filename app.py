import gradio as gr
import os
from groq import Groq
from duckduckgo_search import DDGS

# ======================
# CONFIG
# ======================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Store chats for different users
# Format: { "username": [list of messages] }
user_chats = {}

# ======================
# FUNCTIONS
# ======================

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No information found."
        summary = ""
        for i, r in enumerate(results, 1):
            summary += f"{i}. {r['title']}\n{r['body']}\n\n"
        return summary.strip()
    except Exception as e:
        return f"Search error: {str(e)}"


def format_history(history):
    if not history:
        return "No messages yet."
    text = ""
    for msg in history:
        role = "You" if msg["role"] == "user" else "Lova"
        text += f"**{role}:** {msg['content']}\n\n"
    return text


def login_user(username):
    username = username.strip()
    if not username:
        return "Please enter a name.", "Not logged in", "No messages yet.", ""
    
    if username not in user_chats:
        user_chats[username] = []
    
    return f"Logged in as: {username}", username, format_history(user_chats[username]), ""


def ask_llm(user_text, username):
    if not username or username == "Not logged in":
        return "Please login first with your name.", format_history([])
    
    if not user_text or not user_text.strip():
        return "Please type something.", format_history(user_chats.get(username, []))

    history = user_chats.get(username, [])

    search_keywords = ["who", "what", "when", "where", "stock", "price", "news", "weather", "current", "latest", "today", "kya", "kaun", "kab", "kahan"]
    need_search = any(word in user_text.lower() for word in search_keywords)
    extra_info = search_web(user_text) if need_search else ""

    messages = [
        {
            "role": "system",
            "content": "You are Lova, a helpful AI assistant created by Saksham Jha. You understand English and Hindi. Keep replies short and natural. Use real-world information when provided."
        }
    ]

    for msg in history[-8:]:
        messages.append(msg)

    if extra_info:
        prompt = f"User: {user_text}\n\nReal-world info:\n{extra_info}\n\nAnswer helpfully:"
    else:
        prompt = user_text

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-20b",
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Sorry, error: {str(e)}"

    # Save to this user's history
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    user_chats[username] = history

    return reply, format_history(history)


def process_text(user_text, username):
    reply, history_text = ask_llm(user_text, username)
    return reply, history_text


def clear_chat(username):
    if username and username != "Not logged in":
        user_chats[username] = []
    return "", "Chat cleared."


def logout():
    return "Not logged in", "Not logged in", "No messages yet.", ""


# ======================
# INTERFACE
# ======================

with gr.Blocks(title="Lova AI - Smart Chat Assistant by Saksham Jha") as demo:
    gr.Markdown("""
    # 🎙️ Lova AI
    ### Smart Chat Assistant created by Saksham Jha
    
    **Demo Sign-in:** Enter your name to start.  
    Each name has its own chat history.
    """)

    with gr.Row():
        username_input = gr.Textbox(label="Your Name", placeholder="Type your name here...")
        login_btn = gr.Button("Login", variant="primary")
        logout_btn = gr.Button("Logout")

    login_status = gr.Textbox(label="Status", value="Not logged in", interactive=False)
    current_user = gr.State("Not logged in")  # hidden memory of who is logged in

    history_box = gr.Markdown("No messages yet.")
    
    text_input = gr.Textbox(
        label="Type your message",
        placeholder="Type in English or Hindi..."
    )
    
    reply_box = gr.Textbox(label="Lova's Reply")

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat")

    # Login
    login_btn.click(
        fn=login_user,
        inputs=username_input,
        outputs=[login_status, current_user, history_box, reply_box]
    )

    # Logout
    logout_btn.click(
        fn=logout,
        outputs=[login_status, current_user, history_box, reply_box]
    )

    # Send message
    send_btn.click(
        fn=process_text,
        inputs=[text_input, current_user],
        outputs=[reply_box, history_box]
    )
    text_input.submit(
        fn=process_text,
        inputs=[text_input, current_user],
        outputs=[reply_box, history_box]
    )

    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        inputs=current_user,
        outputs=[reply_box, history_box]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)