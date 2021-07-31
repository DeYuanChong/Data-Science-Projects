#!/usr/bin/env bash

echo "Please Indicate the approach desired: "

echo "1 - Regression of O-level Math Scores"

echo "2 - Multi-Class Classification of O-level Math Scores"

echo "Your Choice: "

read user_input

python3 src/program.py $user_input
