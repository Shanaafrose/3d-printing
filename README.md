# AI-Based Optimization of 3D Printing Parameters for Maximum Tensile Strength

## Project description
This project is a beginner-friendly Streamlit application for a mechanical engineering student project review. It helps a user:

- load experimental data or use an illustrative demo dataset,
- train a Random Forest regressor for tensile strength,
- compare model metrics honestly and without claiming validation beyond the data,
- run a hand-written Genetic Algorithm to search for improved 3D printing parameters,
- download a simple HTML report.

The title and academic metadata are intentionally aligned with the project context in the coursework brief.

## Honesty and limitations statement
This app does not claim that any predicted tensile strength is a validated physical result. All metrics and charts are computed at runtime from the data loaded by the user or from the explicit demo dataset. The optimizer maximizes the model's predicted tensile strength only, and the app clearly warns that printed specimens must still be physically tested.

This project is for academic review and learning. It is not a substitute for a calibrated manufacturing optimization workflow.

## Why these libraries were chosen
- streamlit: easiest way to build a beginner-friendly local web app quickly and clearly.
- pandas: clean, readable CSV handling and dataset inspection.
- numpy: efficient numeric work for the GA and preprocessing.
- scikit-learn: Random Forest regression, preprocessing, cross-validation, and evaluation metrics.
- plotly: interactive and easy-to-read charts for engineering data exploration.
- joblib: lightweight model serialization for saving and loading trained models.

## Folder structure
```text
3d_print_optimizer/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── data/
│   ├── experimental_data.csv   # user places their CSV here
│   └── sample_template.csv      # format example only
├── models/
│   └── (optional saved model files)
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── genetic_algorithm.py
│   ├── visualizations.py
│   ├── report.py
│   ├── demo_data.py
│   └── ui.py
├── tests/
│   └── smoke_test.py
└── .gitignore
```

## Dataset format
The expected CSV columns are:

| Column name | Unit | Description |
| --- | --- | --- |
| layer_height | mm | layer thickness |
| nozzle_temperature | degC | print temperature |
| print_speed | mm/s | nozzle travel speed |
| infill_density | % | infill percentage |
| material_type | categorical | PLA, PETG, ABS or similar |
| tensile_strength | MPa | tensile strength of the specimen |

Notes:
- The template file is only a format example and is not real data.
- Extra columns are ignored after import.
- Duplicate rows are allowed because experimental replicates are legitimate.

## Windows run guide
Run these commands in Command Prompt or PowerShell from the folder where you want the project to live:

```powershell
mkdir 3d_print_optimizer
cd 3d_print_optimizer
code .
python -m venv venv
venv\Scripts\activate
# If PowerShell blocks activation:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
# Place your CSV at: data\experimental_data.csv
streamlit run app.py
```

The app opens at:
- http://localhost:8501

## Troubleshooting
| Issue | Fix |
| --- | --- |
| python not recognized | Install Python and reopen the terminal, or use the full path to python.exe |
| streamlit not recognized | Activate the venv before running the app |
| ModuleNotFoundError | Reinstall dependencies with `pip install -r requirements.txt` |
| Port already in use | Use `streamlit run app.py --server.port 8502` |
| CSV encoding errors | Save the file as UTF-8 and try again |
| DataError: column missing | Check the CSV headers and use the column mapping tool |
| Blank page or cache issue | Press `C` in the browser, then clear the Streamlit cache |
| Slow GA | Lower population size or number of generations |

## Manual testing checklist
- Startup: app launches without errors when data folder is empty.
- Dataset loading: valid CSV, missing file, missing column, bad values handled.
- Model training: model trains on demo and real data.
- Prediction: inside-range and outside-range values are handled safely.
- GA: small and large settings run; fixed material and search-across-materials modes behave correctly.
- Invalid input handling: bad numeric values and invalid bounds are rejected gracefully.
- Comparison charts: render without errors.
- Downloads: CSV and HTML outputs open correctly.
- Demo banner: visible on every page in demo mode and absent in real mode.

## Run smoke test
```powershell
python tests/smoke_test.py
```

This script uses demo data only and prints PASS/FAIL for the key checks.

## Notes for the student
- Keep `SUPERVISOR_NAME = "Enter supervisor name"` as a placeholder until your supervisor confirms the final name.
- Update printer model and material limits in `config.py` before final submission.
- The app is designed to show neutral, beginner-friendly interpretations of metrics instead of claiming physical validation.
