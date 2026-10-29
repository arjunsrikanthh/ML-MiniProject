# Run this project on the Mac

For the live canvas, double-click **Open Live Drawing.command**. Draw on the
canvas at http://127.0.0.1:7862. Both the TensorFlow CNN V1 and the final
50-class XGBoost model predict the same drawing after each completed stroke.
You can turn off automatic predictions and use **Compare models** instead.
**Clear drawing** resets the canvas and predictions. Example buttons are
included for practicing, but your own strokes are processed by the same models.

The updated canvas uses a wider default brush and offers **Match freehand
strokes to the training drawings**. This adjusts stroke thickness and padding
before both models see the same 28x28 image. Turn it off to compare against
the original notebook preprocessing, especially for imported low-resolution
bitmaps. It is not a guarantee of accuracy. The all-category diagnostic and
its separate heldout bitmap controls are recorded in
`verification/canvas_study/REPORT.md`. Original trained weights and reported
benchmark scores have not changed.

This is live prediction using existing trained weights, not live model
training. It supports the 50 classes listed in the app. Unsupported objects
still receive a guess from those classes. Model scores are not calibrated
guarantees of correctness.

For the code-and-results walkthrough:

Double-click **Open Demo.command**. It opens JupyterLab in your browser with
`demonstration/demo.ipynb`, the original demonstration notebook.
The launcher keeps the local notebook server running in the background.
Running the launcher again reopens the same server.

The project has its own Python environment in `.venv/`. Select the notebook
kernel **ML MiniProject (Mac)**. Use **Run > Run All Cells** to train the small
baseline and run the saved CNN and XGBoost models. The large models are already
trained; running this demo does not retrain them.

To try another drawing, save a PNG or JPG in `demonstration/my_drawings/`, then
change `MY_DRAWING` in the notebook and rerun that cell. The CNN and XGBoost
comparison uses that same drawing.

## Files

- `src/`: code from the GitHub repository, with Mac CPU selection.
- `data/processed/`: the supplied datasets and fixed splits.
- `results/`: the supplied scores, charts, tuning records and trained models.
- `demonstration/`: the two notebooks and example images.
- `presentation/`: the slides and PDFs; differing repository copies are retained.
- `guides/`: the supplied setup and presentation preparation PDFs.
- `requirements.txt`: Mac/CPU packages, including JupyterLab.
- `requirements-linux-gpu.txt`: the original NVIDIA environment for reference.

## Warnings and saved logs

The demo notebook was executed on this Mac without error outputs. The training
notebook retains original Linux/NVIDIA outputs, including a failed CUDA
initialization message from a run with no NVIDIA device. A note now identifies
those historical outputs. No full training rerun on the Mac is implied.

Logistic regression and SVM deliberately keep the paper's iteration limits.
Reaching that limit before convergence is recorded in new run metadata and
explained in the console. Other warnings are reported rather than blanket
suppressed. The supplied model weights and historical scores are unchanged.

If this environment needs rebuilding, run **Setup Mac.command** first. It needs
Python 3.12 and Homebrew. XGBoost uses Homebrew's `libomp` runtime. No GPU is
required for this demonstration.

The `pipeline.ipynb` notebook is for training experiments. Its `full` mode can
download raw data and take many hours. Use the demo notebook for presenting.

Assembled from supplied project source, notebooks, datasets, and trained-model
bundles on 2026-10-21.
