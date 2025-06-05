#!/bin/bash

pip install -U packaging setuptools wheel ninja
pip install --no-build-isolation axolotl[flash-attn,deepspeed]
pip install -U -r requirements.txt
