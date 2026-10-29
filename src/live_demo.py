"""Local canvas using the supplied 50-class TensorFlow CNN and XGBoost weights."""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, grey_dilation, grey_erosion
import gradio as gr
import xgboost as xgb

from train_cnn import build

ROOT = Path(__file__).resolve().parents[1]
STROKE_RATIO = 0.06
NORMALIZED_MARGIN = 0.12


def prepare_drawing(value, *, normalize_strokes=False):
    """Prepare a canvas; optional stroke/size matching for QuickDraw training pixels.

    False retains the original notebook preprocessing for comparisons. True
    estimates line width on the high-resolution canvas, adjusts it toward the
    training raster's line-to-shape ratio and reduces excess padding. No model
    weights or training/test data are changed.
    """
    if isinstance(value, dict):
        value = value.get("composite")
    if value is None:
        return None
    image = Image.fromarray(np.asarray(value, dtype=np.uint8))
    if image.mode == "RGBA":
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.size == 0 or float(np.ptp(gray)) < 2:
        return None
    if gray.mean() > 127:
        gray = 255 - gray
    ys, xs = np.nonzero(gray > 0.2 * gray.max())
    if len(ys) == 0:
        return None
    if normalize_strokes:
        mask = gray > 0.2 * gray.max()
        extent = max(int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1)
        distance = distance_transform_edt(mask)
        ridge = (distance == maximum_filter(distance, size=3)) & (distance > 0)
        measured_width = 2 * float(np.median(distance[ridge]))
        radius = round((extent * STROKE_RATIO - measured_width) / 2)
        if radius:
            r = abs(radius)
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            footprint = xx * xx + yy * yy <= r * r
            adjusted = (grey_dilation if radius > 0 else grey_erosion)(gray, footprint=footprint)
            # Keep the original image if erosion would remove the drawing.
            if adjusted.max() > 0:
                gray = adjusted
                ys, xs = np.nonzero(gray > 0.2 * gray.max())
    gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    margin = NORMALIZED_MARGIN if normalize_strokes else 0.2
    side = max(max(gray.shape), int(max(gray.shape) * (1 + margin)))
    square = np.zeros((side, side), np.float32)
    top, left = (side - gray.shape[0]) // 2, (side - gray.shape[1]) // 2
    square[top:top + gray.shape[0], left:left + gray.shape[1]] = gray
    small = np.asarray(Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.Resampling.LANCZOS), np.float32)
    return small * 255 / max(float(small.max()), 1)


class Classifiers:
    def __init__(self):
        with np.load(ROOT / "data/processed/qd_50.npz") as data:
            self.names = data["class_names"].tolist()
        self.cnn = build("V1", len(self.names))
        self.cnn.load_weights(ROOT / "demonstration/models/cnn_v1_50_raw.weights.h5")
        self.xgb = xgb.Booster(model_file=ROOT / "results/xgb_extended/runs/A1/best.ubj")
        self.xgb.set_param({"device": "cpu", "nthread": 4})
        if self.xgb.num_features() != 784:
            raise ValueError("XGBoost input does not match the 28x28 drawings.")

    def predict(self, value, normalize_strokes=True):
        pixels = prepare_drawing(value, normalize_strokes=normalize_strokes)
        if pixels is None:
            return {}, {}, None, "Draw one object to see the models' guesses."
        start = time.perf_counter()
        cnn = self.cnn(pixels.reshape(1, 28, 28, 1), training=False).numpy()[0]
        trees = self.xgb.inplace_predict(pixels.reshape(1, -1))[0]
        elapsed = (time.perf_counter() - start) * 1000
        if not (np.isfinite(cnn).all() and np.isfinite(trees).all()):
            raise ValueError("A model returned invalid predictions.")
        scores = lambda p: {name: float(p[i]) for i, name in enumerate(self.names)}
        a, b = self.names[int(cnn.argmax())], self.names[int(trees.argmax())]
        status = f"CNN: **{a}** · XGBoost: **{b}** · prediction time: **{elapsed:.0f} ms**"
        return scores(cnn), scores(trees), pixels.astype(np.uint8), status


def build_app(classifiers):
    with gr.Blocks(title="Live drawing — CNN and XGBoost") as app:
        gr.Markdown("# Draw a doodle\nDraw one object. The CNN and XGBoost will guess from the same 50 categories.")
        with gr.Row():
            with gr.Column():
                canvas = gr.Sketchpad(
                    label="Draw here", value=np.full((400, 400, 3), 255, np.uint8),
                    type="numpy", image_mode="RGB", canvas_size=(400, 400), fixed_canvas=True,
                    brush=gr.Brush(default_size=20, colors=["#000000"], default_color="#000000", color_mode="fixed"),
                    layers=False,
                )
                with gr.Row():
                    compare = gr.Button("Compare models", variant="primary")
                    clear = gr.Button("Clear drawing")
                automatic = gr.Checkbox(value=True, label="Update predictions after each stroke")
                normalize = gr.Checkbox(
                    value=True, label="Match freehand strokes to the training drawings",
                    info="Helps thin canvas strokes. Try switching it off for drawings imported from small bitmap images.",
                )
                with gr.Row():
                    star = gr.Button("Try example: star")
                    umbrella = gr.Button("Try example: umbrella")
            with gr.Column():
                cnn = gr.Label(label="CNN V1 · top five", num_top_classes=5)
                trees = gr.Label(label="XGBoost · top five", num_top_classes=5)
                preview = gr.Image(label="The 28 × 28 input both models see", type="numpy", height=140, interactive=False)
        status = gr.Markdown("Draw one object to see the models' guesses.")
        gr.Markdown("Predictions run locally on this Mac using previously trained models. "
                    "Both models always choose a supported category; a high score does not guarantee a correct answer.")
        with gr.Accordion("50 supported categories", open=False):
            gr.Markdown(", ".join(sorted(classifiers.names, key=str.lower)))
        outputs = [cnn, trees, preview, status]
        compare.click(classifiers.predict, [canvas, normalize], outputs, api_name="classify", concurrency_limit=1)
        def on_stroke(value, enabled, normalized):
            return classifiers.predict(value, normalized) if enabled else (gr.skip(),) * 4
        canvas.input(on_stroke, [canvas, automatic, normalize], outputs, trigger_mode="always_last", concurrency_limit=1)
        normalize.change(classifiers.predict, [canvas, normalize], outputs, concurrency_limit=1)
        def reset():
            return np.full((400, 400, 3), 255, np.uint8), {}, {}, None, "Draw one object to see the models' guesses."
        clear.click(reset, outputs=[canvas, *outputs], api_name="clear_drawing")
        def example(filename, normalized):
            image = np.asarray(Image.open(ROOT / "demonstration/my_drawings" / filename).convert("RGB"))
            return image, *classifiers.predict(image, normalized)
        star.click(lambda normalized: example("star.png", normalized), inputs=normalize, outputs=[canvas, *outputs], concurrency_limit=1)
        umbrella.click(lambda normalized: example("umbrella.jpg", normalized), inputs=normalize, outputs=[canvas, *outputs], concurrency_limit=1)
    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    models = Classifiers()
    build_app(models).queue().launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False)
