#!/bin/bash -x

python -m venv venv

source ./venv/bin/activate

pip install -r requirements.txt

uvicorn llm_proxy:app --reload --host 0.0.0.0 --port 8000
