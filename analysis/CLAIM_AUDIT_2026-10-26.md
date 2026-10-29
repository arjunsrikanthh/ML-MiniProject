# ML project claim audit - 26 October 2026

The attachments are the same project: both supplied PDFs have identical SHA-256 hashes to the copies under `presentation/` and `guides/`. The experimental numbers have saved supporting evidence. Several explanations in the original slides and preparation guide are more certain than the evidence permits.

This audit inspected saved datasets, all 46 run records and validation confusion matrices, original-author code and Git history, and official framework documentation. It did not retrain, load the large XGBoost model, restart demos, or change historical results. Recalculating a saved confusion matrix verifies its arithmetic and consistency with the run record; it does not independently replay model fitting or prove how a historical file was produced.

Machine-readable recalculations: [evidence.json](evidence.json).

## 1. The 70,000-example SVM claim

**Verdict: strongly supported for the original logged SVM experiments; verified as the size of our supplied diagnostic datasets. The explanation of the full accuracy gap remains incomplete.**

### Original authors' evidence

- All eight SVM confusion matrices in the original README have exactly **7,000 validation examples in every class**. Recalculated diagonal/total accuracies reproduce the paper's SVM figures.
- Their historical loader at commit `42e490e6ae4012f1a69558f487fb0fcbab1ef438` explicitly sets `max_num_examples_per_class = 70000` and uses the 80/10/10 split with seed 42.
- Commit `cfe4189a2bb11bc8db7a7e80d77bda81df38ecec`, dated 21 November 2018, changes that limit to 20,000. Its README already contains the eight SVM matrices matching the final paper. The later loader and final report say 20,000, whereas the retained SVM logs correspond to 70,000.
- Their `svm.py` hard-codes `max_iter=500` for all four kernels and the same explicit RBF/poly/sigmoid parameters used in our implementation.

Sources: [original SVM logs](https://github.com/jervisfm/cs229-project/blob/03cd1a9280319ed08b3d70f3decd5a05155f8ed2/README.md#sklearn-svc), [historical 70k loader](https://github.com/jervisfm/cs229-project/blob/42e490e6ae4012f1a69558f487fb0fcbab1ef438/massageData.py), [70k-to-20k change](https://github.com/jervisfm/cs229-project/commit/cfe4189a2bb11bc8db7a7e80d77bda81df38ecec), [original SVM code](https://github.com/jervisfm/cs229-project/blob/03cd1a9280319ed08b3d70f3decd5a05155f8ed2/svm.py).

**70,000 means total selected examples per class. Training uses 56,000, validation 7,000, and test 7,000.** Calling it 70,000 training examples per class would be incorrect. We have not recovered a run manifest proving the exact commit/environment that produced every published number, but the independent code/history/log evidence strongly supports this data-size interpretation.

### Our saved diagnostic experiments

Both `qd70k_3.npz` and `qd70k_10.npz` contain the expected class names and exactly 56,000/7,000/7,000 labels per class. All eight diagnostic run JSONs specify the same kernels/settings and 500-iteration cap as the corresponding 20k experiments. All 16 SVM validation accuracies were recalculated from saved confusion matrices and agree with their CSV and JSON records within rounding.

| Kernel | Classes | Original logged accuracy | Our 20k total/class | Our 70k total/class | Remaining 70k gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| RBF | 3 | 62.681% | 84.000% | 75.029% | +12.348 points |
| RBF | 10 | 61.013% | 74.625% | 64.337% | +3.324 points |
| Polynomial | 3 | 51.695% | 66.733% | 54.219% | +2.524 points |
| Polynomial | 10 | 50.894% | 73.875% | 56.593% | +5.699 points |

Local evidence: `data/processed/qd70k_{3,10}.npz`, `results/runs/{SVM,svm70k}_*.json`, their `results/figures/*/confusion_dev.npy`, and `results/tables/svm_reported_20k_70k.csv`. Exact numbers and class support are in this audit's JSON.

**What follows:** the score declines under the larger-data protocol are real in the saved artifacts, and move toward the original scores. A hard iteration cap can prevent solver convergence; [SVC documents max_iter as a solver limit](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

**What does not follow:** these runs do not isolate convergence as the sole cause. Changing the sample count also changes training and validation membership, potentially scaling statistics and solver sample ordering. The historical run records do not preserve per-solver iteration counts or fitted SVMs for this audit. The residual 12.35-point three-class RBF gap is substantial. No controlled sample-size-by-iteration-budget experiment or old-library replay was performed. The guide's suggestion that the remainder is explained by library versions is unverified.

Recommended wording: “The original SVM logs and code history support 70,000 selected drawings per class. Our matched-parameter 70k diagnostic runs lowered RBF/poly validation accuracy toward the paper's results. This supports a data-size/iteration-budget explanation, but does not fully explain the remaining differences.”

## 2. The CNN version-selection bug

**Verdict: the bug is verified; its responsibility for each published row is not.**

The original `cnn.py` has independent `if` statements for versions 2/3/4/5, with `else: model()` attached only to the version-5 check. Executing just those original dispatch statements with stub model constructors gives 1->V1, 2->V1, 3->V1, 4->V1, 5->V5. No TensorFlow execution was necessary.

This proves what the inspected script does. It does not prove that every final-paper row was generated with that exact script revision. The differing V3/V4 paper scores also prevent treating all historical labels as conclusively identified.

Recommended wording: “The released source contains an overwrite bug that could explain the V2 discrepancy; the architecture used for every historical reported run is uncertain.”

Source: [original dispatch](https://github.com/jervisfm/cs229-project/blob/03cd1a9280319ed08b3d70f3decd5a05155f8ed2/cnn.py#L329-L338).

## 3. BatchNorm explains the transfer-learning gaps

**Verdict: a technically plausible hypothesis, not a demonstrated causal result.**

The supplied transfer-learning scores agree with saved validation confusion matrices. The original README does show Inception training accuracy near 99.99% and validation near 45.7%. Modern Keras explicitly runs frozen BatchNormalization in inference mode. Historical Keras 2.2.4 source selects the normalization branch through the training phase separately from freezing state updates, supporting the proposed mechanism.

However, the original run's exact package versions and numerical execution have not been reconstructed; no controlled old/new BatchNorm experiment reproduces the gap. A train/validation gap is not unique evidence of BatchNorm failure. Preprocessing, training choices, seeds and other implementation differences remain possible contributors. VGG19 agreement is useful supporting evidence, not proof that the whole pipeline is identical.

Sources: [modern Keras semantics](https://keras.io/api/layers/normalization_layers/batch_normalization/), [Keras 2.2.4 normalization implementation](https://github.com/keras-team/keras/blob/2.2.4/keras/layers/normalization.py), [old freezing of updates](https://github.com/keras-team/keras/blob/2.2.4/keras/engine/base_layer.py), [original training logs](https://github.com/jervisfm/cs229-project/blob/03cd1a9280319ed08b3d70f3decd5a05155f8ed2/README.md).

Recommended wording: “Framework-dependent frozen BatchNorm behavior is a plausible contributor; we did not establish it with a controlled replay.” Remove claims that the paper's architectural conclusion was proven to be an artefact.

## 4. Accuracy and timing claims

| Claim | Verdict and appropriate scope |
| --- | --- |
| The recorded model results exist | Verified: all 46 ledger validation scores match the associated saved confusion matrices within rounding. The ledger contains 3 LR, 8 SVM, 18 CNN, 4 transfer-learning, 8 extra 70k SVM and 5 XGBoost records. |
| CNN gets 81.2% on 50 classes | Correct as rounded validation accuracy: 81.21%. Cached test labels/predictions independently recalculate to 81,071/100,000 = 81.071%. Use 81.1% for that test result. Neither measures arbitrary new freehand drawing accuracy. |
| Final XGBoost gets about 78% | Correct for recorded validation 77.95%; historical test is recorded as 77.89%, with an earlier bounded checkpoint audit recording exact 77.886%. No new XGBoost inference was performed here. |
| CNN wins on 50 classes | Supported among the tested configurations on this dataset/protocol. It is not a universal comparison of CNNs against every possible tree-based approach. |
| CNN is 16.7 times faster | The selected historical fit-time ratio is 7,191.10/431.43 = 16.668. This excludes hyperparameter-search cost and is not current-Mac inference speed or a hardware-independent guarantee. Timing definitions differ from the original paper, so cross-paper speed claims need qualification. |
| Same split for all main model families | The supplied implementation loads the shared files for each class set. The 70k SVM diagnostic deliberately uses different files. This does not prove an exact cross-era replay or perfect deduplication/person-independent splits. |
| Transfer feature caching gives an identical result | Exact equivalence requires a deterministic frozen feature mapping and matching preprocessing/precision. The implementation includes modern resize behavior and float16 cached mixed8 features for Inception fine-tuning, so bit-identical historical execution is not established. |

## 5. XGBoost search, plateau and test-set claims

**Verified saved records:** Optuna selects settings using validation accuracy; early stopping monitors validation log-loss. A1 continues from 1,168 rounds, trains to 2,243, and retains the best 2,193 rounds. Its total fit time is 7,191.10 seconds. Three 50-class settings reach early stopping at validation accuracies 77.95%, 77.95% and 77.77%.

**Correction:** the fourth extended setting A2-3 is explicitly `CAPPED_BY_BUDGET`, not fully converged (77.43%). “Every extended run trained fully with no time limit” is false. The selected final model did converge under the stated early-stopping rule, within a finite search/round budget.

**Correction:** these trials show a plateau around 78% for the configurations tried. They do not establish a fundamental XGBoost/model-family ceiling. Depth was limited to 9, settings were selected through the 10-class search rather than an independent full 50-class search, and repeated seeds/alternative features remain unexplored. “The limit is definitely the model rather than tuning” is too strong.

**Correction:** “the test set was used once” is not literally true across the complete project. The ledger contains initial and extended finalist test scores for 10 and 50 classes; later independent checks also reevaluate saved models. The inspected tuning code uses validation, not test, to select settings. Say: “Selection used validation; separate stage finalists were evaluated on test.” Multiple test evaluations alone do not prove test-driven tuning.

**Correction:** raw-pixel trees can learn feature interactions. “Trees see independent pixels” is misleading if it implies they cannot combine pixels. CNNs explicitly share local filters, giving a useful spatial inductive bias, but the causal reason for this measured gap has not been isolated. The paired 50-class error analysis is still incomplete.

Source records: `results/xgb_extended/runs/{A1,A2-1,A2-2,A2-3}/result.json`, their round curves, `results/runs/XGB_*.json`, and `src/train_xgboost.py`.

Recommended wording: “The tested converged raw-pixel XGBoost settings cluster around 78% validation accuracy, below our best CNN. Wider feature/parameter/seed experiments would be needed to establish why or whether that limit can be improved.”

## 6. Which materials and deliverables are complete

- Both attached PDFs match the older local presentation and preparation guide byte-for-byte. Ignoring names, they describe this 3/10/50-class TensorFlow project, not the separate older DrawSearch experiment.
- The corrected local deliverables are under `delivery/ML_Presentation_Package/`: eight-slide PPTX, eight-page PDF backup, two-page course PDF and source-only ZIP. All four still match their previously verified hashes. The full package has 583 entries and passed a fresh archive CRC check.
- The saved demo has 10 executed code cells and no saved errors; a separate fresh-kernel execution record passed on 22 October. Dependencies pass `pip check` now and all eight source modules parse. This audit did not rerun the notebook or canvas because the user's later stop instruction keeps both demos and heavy XGBoost loading stopped.
- The later paired CNN/XGBoost test error comparison is incomplete: CNN cache is available; final-XGBoost row-level predictions, paired category/error counts and comparison figures remain pending. This is additional analysis, separate from completing the original model experiments.
- At the time of this audit, the private faculty/TA repository handoff was unverified. The project source is now published in a public repository; evaluator access to the companion artifacts and formal submission remain separate steps.
- The faculty email and its two-page guideline attachment require a private shared repository with README, short PDF, slides, live demo, explanation/Q&A and individual contribution. The prep guide's eight-slide plan, 10-minute slot and mandatory three-minute video are not requirements in those official sources. No video was found in this project folder, but the live demo is the stated official requirement. Actual rehearsal, contribution agreement and a current review/submission schedule remain unverified.

Use the corrected delivery deck for its more careful claims. The older guide still needs qualification before its explanations are repeated as proven facts. No original PDFs, notebooks, datasets, model weights or historical records were edited by this audit.
