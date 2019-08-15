#!/bin/bash
black resources
pdoc3 --force --html -o docs resources
cp -rf docs/resources/* docs
rm -rf docs/resources