#!/usr/bin/env bash


rm -rf ./facenet_pytorch.egg-info ./dist ./build
pip3 install wheel
python3 setup.py sdist bdist_wheel
pip3 uninstall -y facenet-pytorch
pip3 install ./dist/facenet_pytorch-*-py3-none-any.whl
