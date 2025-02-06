import gradio as gr


def compute_shape(image):
    return f"Shape {image.shape}"


with gr.Blocks() as demo:
    with gr.Row():
        img = gr.Image()
        out = gr.Textbox()
    img.change(compute_shape, img, out)


try:
    demo.launch(server_port=9677)
except OSError as e:
    if "address already in use" in str(e):
        print("Port 9000 is already in use.  Trying a different port.")
