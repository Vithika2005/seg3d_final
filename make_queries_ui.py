"""Click-to-label query builder in the browser (use instead of make_queries.py).
1) pick an image  2) type ONE query  3) click the correct object in the image  4) press Add.
Saves to queries.json (adds to it; delete the file to start over). Usage: python make_queries_ui.py"""
import glob, json, os
import gradio as gr
from PIL import Image, ImageDraw

OUT = "queries.json"


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else []


def table():
    return [[i + 1, d["image"], d["query"], d.get("tag", ""), str(d["click"])] for i, d in enumerate(load())]


def show(path):
    return (Image.open(path).convert("RGB") if path else None), None, "Now click the correct object after typing your query."


def on_click(path, evt: gr.SelectData):
    x, y = int(evt.index[0]), int(evt.index[1])
    im = Image.open(path).convert("RGB"); d = ImageDraw.Draw(im); r = max(6, im.width // 100)
    d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=3)
    return im, [x, y], f"Click recorded at [{x}, {y}] - press Add."


def add(path, query, oov, click):
    query = (query or "").strip().lstrip("!").strip()
    if not path:
        raise gr.Error("Pick an image first.")
    if not query:
        raise gr.Error("Type a query first.")
    if not click:
        raise gr.Error("Click the correct object in the image first.")
    data = load(); data.append({"image": path, "query": query, "click": click, "tag": "oov" if oov else "in_vocab"})
    json.dump(data, open(OUT, "w"), indent=2)
    return table(), "", None, f"Saved ({len(data)} queries). Type the next query, then click its answer.", Image.open(path).convert("RGB")


def delete_last():
    data = load()
    if data:
        data.pop(); json.dump(data, open(OUT, "w"), indent=2)
    return table(), f"{len(data)} queries left."


with gr.Blocks(title="Query builder") as demo:
    click_state = gr.State(None)
    gr.Markdown("## Query builder\nExamples: `the chair to the left of the table`, `closest lamp`, `what is on top of the table`, "
                "`what is between the sofa and the window`, `something you can sit on`. Tick the box if the target object is **not** in your detector vocabulary.")
    path = gr.Dropdown(sorted(glob.glob("samples/*.jpg")), label="Image")
    img = gr.Image(type="pil", interactive=False, label="Click the correct object")
    query = gr.Textbox(label="Query (one per entry)")
    oov = gr.Checkbox(False, label="Target is NOT in the detector vocabulary (out-of-vocabulary)")
    status = gr.Markdown()
    with gr.Row():
        add_btn = gr.Button("Add query", variant="primary"); del_btn = gr.Button("Delete last")
    tbl = gr.Dataframe(value=table, headers=["#", "image", "query", "tag", "click"], interactive=False)
    path.change(show, path, [img, click_state, status])
    img.select(on_click, path, [img, click_state, status])
    add_btn.click(add, [path, query, oov, click_state], [tbl, query, click_state, status, img])
    del_btn.click(delete_last, None, [tbl, status])

if __name__ == "__main__":
    demo.launch(server_port=7861)
