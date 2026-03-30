# Flexible Evaluation Pipeline

Pipeline nay duoc thiet ke de:

- thay model lien tuc
- bo sung prompt lien tuc
- bo sung dataset lien tuc, moi dataset co extractor va metric rieng
- chon tap model va tap dataset moi lan inference
- ho tro ca local path va Hugging Face Hub
- luu report day du, khong cat bot response hay prompt
- chay inference bang `transformers` voi batch size tuy chinh
- giu code inference chinh o notebook, con logic tuy chinh o module Python
- dung duoc tren local, Google Colab, Kaggle va Run.ai
- toi uu kha nang chay lai ket qua giong nhau

## Cau truc thu muc

```text
.
|-- notebooks/
|   `-- run_first_experiment.ipynb
|-- outputs/
|-- src/
|   `-- eval_pipeline/
|       |-- __init__.py
|       |-- dataset_adapters.py
|       |-- evaluator.py
|       |-- metrics.py
|       |-- model_runner.py
|       |-- registry.py
|       |-- reporting.py
|       |-- specs.py
|       `-- utils.py
`-- requirements.txt
```

## Cach dung

1. Cai dat goi can thiet:

   ```bash
   pip install -r requirements.txt
   ```

2. Mo [`notebooks/run_first_experiment.ipynb`](/c:/Users/Admin/Documents/eval_model_pipeline/notebooks/run_first_experiment.ipynb).

3. Sua:

- `SELECTED_MODELS`
- `SELECTED_DATASETS`
- `BATCH_SIZE`
- `MAX_NEW_TOKENS`
- `LOCAL_MODEL_OVERRIDES`
- `LOCAL_DATASET_OVERRIDES`
- `SYSTEM_PROMPT_OVERRIDES`

4. Chay notebook.

## Cach them model moi

Them `ModelSpec` moi trong [`src/eval_pipeline/registry.py`](/c:/Users/Admin/Documents/eval_model_pipeline/src/eval_pipeline/registry.py).

Model co the co:

- `local_paths`: danh sach duong dan local theo tung moi truong
- `hf_repo_id`: repo tren Hugging Face, dung khi local path khong ton tai

## Cach them dataset moi

1. Tao adapter moi trong [`src/eval_pipeline/dataset_adapters.py`](/c:/Users/Admin/Documents/eval_model_pipeline/src/eval_pipeline/dataset_adapters.py).
2. Dang ky `DatasetSpec` moi trong [`src/eval_pipeline/registry.py`](/c:/Users/Admin/Documents/eval_model_pipeline/src/eval_pipeline/registry.py).
3. Adapter tu quyet dinh:

- cach load dataset
- cach render prompt
- cach extract prediction
- cach tinh metric
- cach tong hop summary

## Report duoc luu o dau

Moi lan chay tao mot thu muc moi trong `outputs/runs/<timestamp>_<run_name>/`:

- `manifest.json`: cau hinh run
- `environment.json`: thong tin phien ban va git commit
- `overall_summary.json`: tong hop tat ca cap model/dataset
- `summary.csv`: bang tong hop
- `pairs/<model>__<dataset>/metrics.json`: metric cua cap do
- `pairs/<model>__<dataset>/samples.jsonl`: tung mau, full prompt, full response, ground truth, prediction, metric

## Tai lap ket qua

Pipeline da:

- set seed cho Python, NumPy, PyTorch, Transformers
- tat `do_sample`
- luu generation config
- luu source path/repo cua model va dataset
- luu package versions va git commit

Luu y: mot so kernel CUDA va driver van co the tao sai khac nho giua cac may.
