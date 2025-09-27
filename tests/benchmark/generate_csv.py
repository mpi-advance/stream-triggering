#!/usr/bin/env python3

from pathlib import Path

import argparse
import csv
import os

reset = "\033[0m"
blue  = "\033[94m"

default_input="../scratch/output"
default_output_dir="../scratch/csv/"

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Top-level directory to look at"
    )

    parser.add_argument(
        "-o"
        "--outfile",
        type=str,

        dest="outfile",
        help="Where to store the output"
    )

    return parser.parse_args()

def main():
    # Expects results after grep
    cmdline_options = parse_arguments()

    if cmdline_options.dir is None:
        print(f'No input directory given -{blue} Using:{reset}{default_input}')
        dir_to_search = Path(default_input)
    else:
        dir_to_search = Path(cmdline_options.dir)

    if cmdline_options.outfile is None:
        default_output=default_output_dir+"data.csv"
        number=2
        while(os.path.exists(default_output)):
            default_output=default_output_dir+"data"+str(number)+".csv"
            number+=1
        print(f'No output file given -{blue} Using:{reset}{default_output}')
    else:
        default_output = cmdline_options.outfile

    for entry in dir_to_search.iterdir():
        if entry.is_file():
            print(f"{blue}Found: {reset}{entry}")
            if "TIOGA" in entry.name:
                device = "MI250"
            elif "TUO" in entry.name:
                device = "MI300"
            else:
                device = "?"

            with open(entry, 'r') as file, open(default_output, 'a') as output:
                writer=csv.writer(output, delimiter=",")
                test_search = True
                for line in file:
                    if "Test:" in line:
                        if not test_search:
                            raise ValueError("Expceted to find time -- found another test instead")
                        
                        test_search = False
                        _, test, num_iters, buff_size = line.strip().split(" ")
                    elif "0 is done:" in line:
                        if test_search:
                            raise ValueError("Expceted to find test -- found a time value instead")
                        
                        _, time = line.strip().split(":")
                            
                        writer.writerow([device, test, num_iters, buff_size, time.strip()])
                        test_search = True

if __name__ == "__main__":
    main()