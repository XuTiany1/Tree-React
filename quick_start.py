import argparse
from model.tot.tree import solve
from tasks.hotpotqa import hotpotQATask
import re

import openai
print(openai.__version__)

args = argparse.Namespace( 
    backend='gpt-3.5-turbo-0125', 
    temperature=0.7, 
    task='hotpotqa', 
    naive_run=False, 
    prompt_sample=None, 
    method_generate='propose', 
    method_evaluate='value', 
    method_select='greedy', 
    react_search=True,
    n_generate_sample=1, 
    n_evaluate_sample=3, 
    n_select_sample=2)


task = hotpotQATask()
ys, infos = solve(args, task, 1)



# Printing results
print("====================================================")
print("====================================================")
print("====================================================")
print("====================================================")
print("\n")
print(f"Model prediction: {ys[0]}")
print("\n")
print("====================================================")
print("\n")
output = task.test_output(idx=3, output="")
print(f"Ground truth: {output}")
print("\n")
print("====================================================")
print("\n")
print("\n")
print("====================================================")
print("====================================================")
print("====================================================")

