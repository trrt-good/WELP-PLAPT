import warnings



import argparse
import json
import csv
import os
from plapt import Plapt
warnings.filterwarnings("ignore")
def write_json(results, filename):
    with open(filename, 'w') as json_file:
        json.dump(results, json_file)

def write_csv(results, filename):
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for result in results:
            writer.writerow([result])

def determine_format_and_update_filename(output_arg, format_arg):
    if output_arg:
        _, ext = os.path.splitext(output_arg)
        if ext not in [".csv", ".json"]:
            output_arg += f".{format_arg or 'json'}"
        return output_arg, (format_arg or "json" if not ext else ext[1:])
    return None, "json"

def main():
    parser = argparse.ArgumentParser(description="Predict affinity using Plapt.")
    parser.add_argument("-t", "--target", nargs="+", required=True, help="The target protein sequence")
    parser.add_argument("-m", "--smiles", nargs="+", required=True, help="List of SMILES strings")
    parser.add_argument("-o", "--output", help="Optional output file path")
    parser.add_argument("-f", "--format", choices=["json", "csv"], help="Optional output file format; required if output is specified without an extension")

    args = parser.parse_args()

    plapt = Plapt()
    if len(args.target) == len(args.smiles):
        results = plapt.predict_affinity(args.target, args.smiles)
    else:
        results = plapt.score_candidates(args.target, args.smiles)

    args.output, output_format = determine_format_and_update_filename(args.output, args.format)

    if args.output:
        if output_format == "json":
            write_json(results, args.output)
        elif output_format == "csv":
            write_csv(results, args.output)
        print(f"Output written to {args.output}")
    else:
        print(results)

if __name__ == "__main__":
    main()
