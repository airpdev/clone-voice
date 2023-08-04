from webui.ui.ui import create_ui
import gradio


def launch_webui():
    template_response_original = gradio.routes.templates.TemplateResponse

    # Magic monkeypatch
    import webui.extensionlib.extensionmanager as em
    scripts = ''.join([f'<script type="module" src="file={s}"></script>' for s in ['scripts/script.js'] + em.get_scripts()])

    def template_response(*args, **kwargs):
        res = template_response_original(*args, **kwargs)
        res.body = res.body.replace(b'</body>',
                                    f'{scripts}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response

    import webui.extensionlib.callbacks as cb
    cb.get_manager('webui.init')()

    create_ui("gradio/soft").queue().launch(share=True,
                                         server_name='0.0.0.0',
                                         server_port=None,
                                         favicon_path='assets/logo.png')
