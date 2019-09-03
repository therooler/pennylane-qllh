#!/bin/bash
black rockyraccoon
pdoc3 --force --html -o docs rockyraccoon
cp -rf docs/rockyraccoon/* docs
rm -rf docs/rockyraccoon