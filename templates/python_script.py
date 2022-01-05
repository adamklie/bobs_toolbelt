# -*- coding: utf-8 -*-

"""
Python script for...
TODO: 
    1. 
    2. 
    3. 
"""

# Built-in/Generic Imports
import glob
import os
import argparse
import warnings


# Libs
import pandas as pd
import numpy as np
# Tags
__author__ = "Adam Klie"
__data__ = "MM/DD/YYYY"


def main(args):
    
    print("Script Title\n" + "-"*len("Script Title"))
    
    #Step X: Load data
    print("Loading...")

    #Step X: Do stuff to data
    print("Calculate...")
    
    #Step X: Save new data
    print("Saving...")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", type=str, help="Example input argument")
    parser.add_argument("--int", type=int, default=10, help="Example int argument")
    parser.add_argument("--out", type=str, default="./", help="Example out argument")
    args = parser.parse_args()
    main(args) 