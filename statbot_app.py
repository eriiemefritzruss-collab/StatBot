import os
import socket

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from starlette.templating import Jinja2Templates
from front_end.js import js
from front_end.css import css
from session_manager import SessionManager
from utils.utils import to_absolute_path


def patch_starlette_template_response():
    original_template_response = Jinja2Templates.TemplateResponse

    if getattr(original_template_response, "_statbot_compat_patch", False):
        return

    def compat_template_response(self, *args, **kwargs):
        # Gradio 4.44.x still uses the older Starlette call shape:
        # TemplateResponse(name, context, ...).
        # Newer Starlette expects TemplateResponse(request, name, context, ...).
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            context = args[1]
            request = context.get("request")
            if request is not None:
                args = (request, args[0], context, *args[2:])
        return original_template_response(self, *args, **kwargs)

    compat_template_response._statbot_compat_patch = True
    Jinja2Templates.TemplateResponse = compat_template_response


def find_available_port(preferred_port, host="127.0.0.1", attempts=20):
    for offset in range(attempts):
        port = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return None

def launch_app():
    patch_starlette_template_response()
    session_manager = SessionManager(config_path="config.yaml")

    def get_bot(request: gr.Request):
        return session_manager.get(getattr(request, "session_hash", None))

    def add_file(files, request: gr.Request):
        return get_bot(request).add_file(files)

    def chat_streaming(message, chat_history, code=None, request: gr.Request = None):
        return get_bot(request).chat_streaming(message, chat_history, code)

    def stream_workflow(chat_history, code=None, request: gr.Request = None):
        yield from get_bot(request).conv.stream_workflow(chat_history, code)

    def open_board(request: gr.Request):
        return get_bot(request).open_board()

    def rendering_code(request: gr.Request):
        return get_bot(request).rendering_code()

    def export_code_action(request: gr.Request):
        return get_bot(request).export_code()

    def down_notebook_action(request: gr.Request):
        return get_bot(request).down_notebook()

    def generate_report_action(chat_history, request: gr.Request):
        return get_bot(request).generate_report(chat_history)

    def down_report_action(request: gr.Request):
        return get_bot(request).down_report()

    def save_dialogue(chat_history, request: gr.Request):
        return get_bot(request).save_dialogue(chat_history)

    def clear_all(message, chat_history, request: gr.Request):
        return get_bot(request).clear_all(message, chat_history)

    def update_config(conv_model, programmer_model, inspector_model, api_key,
                      base_url_conv_model, base_url_programmer, base_url_inspector,
                      max_attempts, max_exe_time,
                      load_chat, chat_history_path, request: gr.Request):
        return get_bot(request).update_config(
            conv_model, programmer_model, inspector_model, api_key,
            base_url_conv_model, base_url_programmer, base_url_inspector,
            max_attempts, max_exe_time, load_chat, chat_history_path
        )

    statbot = session_manager.get("bootstrap")
    with gr.Blocks(theme=gr.themes.Soft(), css=css, js=js) as demo:

        with gr.Tab("statbot"):
            gr.HTML("<H1>Welcome to statbot! Easy Data Analysis!</H1>")
            chatbot = gr.Chatbot(value=statbot.conv.chat_history_display, height=600, label="statbot", show_copy_button=True, type="tuples")
            with gr.Group():
                with gr.Row(equal_height=True):
                    upload_btn = gr.UploadButton(label="Upload Data", scale=1)
                    msg = gr.Textbox(show_label=False, placeholder="Sent message to LLM", scale=6, elem_id="chatbot_input")
                    submit = gr.Button("Submit", scale=1)
                upload_status = gr.Markdown("", visible=True)
            with gr.Row(equal_height=True):
                board = gr.Button(value="Show/Update DataFrame", elem_id="df_btn", elem_classes="df_btn")
                export_notebook = gr.Button(value="Notebook")
                down_notebook = gr.DownloadButton("Download Notebook", visible=False)
                generate_report = gr.Button(value="Generate Report")
                down_report = gr.DownloadButton("Download Report", visible=False)

                edit = gr.Button(value="Edit Code", elem_id="ed_btn", elem_classes="ed_btn")
                save = gr.Button(value="Save Dialogue")
                clear = gr.ClearButton(value="Clear All")
            report_status = gr.Markdown("", visible=True)

            with gr.Group():
                with gr.Row(visible=False, elem_id="ed", elem_classes="ed"):
                    code = gr.Code(label="Code", scale=6)
                    code_btn = gr.Button("Submit Code", scale=1)
            code_btn.click(fn=chat_streaming, inputs=[msg, chatbot, code], outputs=[msg, chatbot]).then(
                stream_workflow, inputs=[chatbot, code], outputs=chatbot)

            df = gr.Dataframe(visible=False, elem_id="df", elem_classes="df")

            upload_btn.upload(fn=add_file, inputs=upload_btn, outputs=upload_status)
            msg.submit(chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
                stream_workflow, chatbot, chatbot
            )
            submit.click(chat_streaming, [msg, chatbot], [msg, chatbot], queue=False).then(
                stream_workflow, chatbot, chatbot
            )
            board.click(open_board, inputs=[], outputs=df)
            edit.click(rendering_code, inputs=None, outputs=code)
            export_notebook.click(export_code_action, inputs=None, outputs=[export_notebook, down_notebook])
            down_notebook.click(down_notebook_action, inputs=None, outputs=[export_notebook, down_notebook])
            generate_report.click(generate_report_action, inputs=[chatbot], outputs=[generate_report, down_report, report_status])
            down_report.click(down_report_action, inputs=None, outputs=[generate_report, down_report, report_status])
            save.click(save_dialogue, inputs=chatbot)
            clear.click(fn=clear_all, inputs=[msg, chatbot], outputs=[msg, chatbot])

        # The Configuration Page
        with gr.Tab("Configuration"):
            gr.Markdown("# System Configuration for statbot")
            with gr.Row():
                conv_model = gr.Textbox(value=statbot.config.get("conv_model", "qwen-max"), label="Conversation Model")
                programmer_model = gr.Textbox(value=statbot.config.get("programmer_model", "qwen-max"), label="Programmer Model")
                inspector_model = gr.Textbox(value=statbot.config.get("inspector_model", "qwen-max"), label="Inspector Model")
            
            api_key = gr.Textbox(value=statbot.config.get("api_key", ""), label="API Key", type="password", placeholder="Input Your API key")
            with gr.Row():
                base_url_conv_model = gr.Textbox(value=statbot.config.get("base_url_conv_model", "https://dashscope.aliyuncs.com/compatible-mode/v1"), label="Base URL (Conv Model)")
                base_url_programmer = gr.Textbox(value=statbot.config.get("base_url_programmer", "https://dashscope.aliyuncs.com/compatible-mode/v1"), label="Base URL (Programmer)")
                base_url_inspector = gr.Textbox(value=statbot.config.get("base_url_inspector", "https://dashscope.aliyuncs.com/compatible-mode/v1"), label="Base URL (Inspector)")

            with gr.Row():
                max_attempts = gr.Number(value=5, label="Max Attempts", precision=0)
                max_exe_time = gr.Number(value=18000, label="Max Execution Time (s)", precision=0)
            with gr.Row():            
                load_chat = gr.Checkbox(value=False, label="Load from Cache")
                chat_history_path = gr.Textbox(label="Chat History Path", visible=False, interactive=True)
                
            save_btn = gr.Button("Save Configuration", variant="primary")
            status_output = gr.Markdown("")
            
            def toggle_chat_history_path(load_chat_checked):
                return gr.Textbox(visible=load_chat_checked, interactive=True)
            
            save_btn.click(
                fn=update_config,
                inputs=[
                    conv_model, programmer_model, inspector_model, api_key,
                    base_url_conv_model, base_url_programmer, base_url_inspector,
                    max_attempts, max_exe_time,
                    load_chat, chat_history_path
                ],
                outputs=[status_output, chatbot]
            )

            load_chat.change(
                fn=toggle_chat_history_path,
                inputs=load_chat,
                outputs=chat_history_path
            )

    configured_port = int(statbot.config.get("server_port", 8000))
    selected_port = find_available_port(configured_port, host=statbot.config.get("server_name", "127.0.0.1"))
    if selected_port is None:
        selected_port = configured_port
    elif selected_port != configured_port:
        print(f"[WARN] Port {configured_port} is busy. Switching to port {selected_port}.")

    launch_kwargs = {
        "server_name": statbot.config.get("server_name", "0.0.0.0"),
        "server_port": selected_port,
        "allowed_paths": [to_absolute_path(statbot.config["project_cache_path"])],
        "share": statbot.config.get("share", False),
        "inbrowser": statbot.config.get("inbrowser", False),
        "show_error": True,
    }

    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise

        print("[WARN] Gradio localhost accessibility check failed. Retrying in local-only mode...")
        demo.launch(**launch_kwargs, _frontend=False)


if __name__ == '__main__':
    launch_app()
