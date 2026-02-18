#!/bin/bash
# cleanup
rm -rf build/
rm -rf dist/
rm -rf turkicnlp.egg-info/
# build
python -m build
