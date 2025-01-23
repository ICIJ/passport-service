#!/bin/bash
N_PROCESSING_WORKERS=
uv run python -m icij_worker workers start -g preprocessing -n "${N_PROCESSING_WORKERS:-1}" passport_service.app.app
