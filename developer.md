# Instructions for Developers

## Environment Setup

Minimal conda environment with all the dependencies of `odc-geo` and all the
linting tools is provided in `dev-env.yml` file.

```
mamba env create -f dev-env.yml
conda activate odc-geo
pip install -e .
```


## VScode Workspace Configuration

```json
{
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": [
        "--ignore=E731,W503",
        "--max-line-length=120"
    ],
    "python.linting.mypyEnabled": true,
    "python.analysis.typeCheckingMode": "off",
    "python.linting.pydocstyleEnabled": false,
    "python.linting.pydocstyleArgs": [
        "--max-line-length=120"
    ],
    "python.linting.pycodestyleEnabled": true,
    "python.linting.pycodestyleArgs": [
        "--max-line-length=120"
    ],
    "python.linting.pylintEnabled": true,
    "python.sortImports.args": [
        "--profile=black"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "-s",
        "-v"
    ]
}
```
