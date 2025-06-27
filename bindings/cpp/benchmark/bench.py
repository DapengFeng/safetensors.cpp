#!/bin/python

from safetensors import safe_open
import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark safetensors file")
    parser.add_argument("path", type=str, help="Path to the safetensors file")
    args = parser.parse_args()

    start_time = datetime.datetime.now()

    results = {}
    with safe_open(args.path, framework="pt") as f:
        for key in f.keys():
            results[key] = f.get_tensor(key)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"Benchmark completed in {duration}")
