#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
from pathlib import Path

ARROW = '\033[0;34m--> \033[0m'

def load_modules_from_file(filepath):
    """Reads a list of modules from a text file."""
    modules = []
    if not os.path.exists(filepath):
        print(f"Warning: Module file '{filepath}' not found. No modules will be loaded.")
        return modules
    
    with open(filepath, 'r') as f:
        for line in f:
            # Strip whitespace and ignore empty lines or comments
            line = line.strip()
            if line and not line.startswith('#'):
                modules.append(line)
    return modules

def compile_test(program, output, defines, modules, st_base_path):
    """Constructs and executes the compilation command."""
    print(f"{ARROW}{output}")
    # Mimic Bash behavior: if d_path is empty, this resolves to /include and /lib
    st_inc = f"-I{st_base_path}/include" if st_base_path else ""
    st_lib = f"-L{st_base_path}/lib" if st_base_path else ""
    libs = "-lstream-triggering" if st_base_path else ""

    # Determine compiler
    compiler="CC"
    extras=[]
    for module in modules:
        if "rocmcc" in module:
            compiler="mpiamdclang++"
            extras=["--offload-arch=gfx942", "-mllvm -amdgpu-early-inline-all=true",
                    "-mllvm -amdgpu-function-calls=false", "-fhip-new-launch-api",
                    "--driver-mode=g++" ]
            break

    # Base compiler command
    cc_cmd = [
        f"{compiler}", "-D__HIP_PLATFORM_AMD__", "-O3", "-g", "-std=c++20", "-x", "hip"
    ]
    cc_cmd.extend(extras)
    
    # Add defines and other arguments
    cc_cmd.extend(defines.split())
    cc_cmd.extend([program, f"{st_inc}", f"{st_lib}", "-o", output, libs])

    cc_str = " ".join(cc_cmd)

    # Wrap the compilation with module loads if they exist
    # (HPC systems typically need this run in the same shell step)
    if modules:
        mod_str = "module load " + " ".join(modules)
        full_cmd = f"{mod_str} && {cc_str}"
    else:
        full_cmd = cc_str

    print(f"+ {full_cmd}")
    
    # Execute the command (using bash login shell to ensure 'module' command is available)
    try:
        subprocess.run(full_cmd, shell=True, executable='/bin/bash', check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Compilation failed for {output}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Wrapper script to help compile ST-related codes on Cray systems")
    parser.add_argument('-f', '--file', required=True, help="Compile file provided.")
    parser.add_argument('-g', action='store_true', help="Add defines for cuda GPUs.")
    parser.add_argument('-X', action='store_true', help="No CXI build")
    parser.add_argument('-H', action='store_true', help="No HIP build")
    parser.add_argument('-C', action='store_true', help="No CUDA build")
    parser.add_argument('-T', action='store_true', help="No THREAD build")
    parser.add_argument('-S', default="", help="Path to ST installation")
    parser.add_argument('-m', '--modlist', help="Use a custom file to specify which modules to load.")

    args = parser.parse_args()

    program_path = Path(args.file)

    if not program_path.exists():
        print(f"Could not find: {args.file}. Stopping.")
        sys.exit(1)

    # Determine device suffix
    cluster = os.getenv("LCSCHEDCLUSTER") or os.getenv("LMOD_SYSTEM_NAME")
    if not cluster:
        raise RuntimeError("Unable to determine cluster from environment variables, stopping.")
    print(f" -> Compiling for: {cluster}")
    device = f"_{cluster}"

    if args.modlist:
        module_path = Path(args.modlist)
    else:
        module_path = Path(f"../install_setup/{cluster.lower()}_modules.txt")
    
    if not module_path.exists():
        print(f"Could not find: {args.modlist}. Stopping.")
        sys.exit(1)


    # Read modules
    modules = load_modules_from_file(module_path)

    # Setup directories
    scratch = Path("scratch")
    dirs = {
        "scratch": scratch,
        "executables": scratch / "execs",
        "flux": scratch / "flux",
        "outputs": scratch / "output",
        "temp": scratch / "tmp"
    }

    for name, path in dirs.items():
        if not path.is_dir():
            print(f" -> Making {name} directory")
            path.mkdir(parents=True, exist_ok=True)

    # Base output path
    base_output = dirs["executables"] / f"{program_path.stem}{device}"
    print(f" -> {args.file} -> {base_output}")

    # Determine default device define
    device_define = "-DNEED_CUDA" if args.g else "-DNEED_HIP"

    # Compilation Steps
    if args.S:
        # Meaning Stream Triggering is Needed
        if not args.X:
            output = f"{base_output}_cxi-coarse"
            defines = f"-DCXI_BACKEND {device_define}"
            compile_test(args.file, output, defines, modules, args.S)

            output = f"{base_output}_cxi-fine"
            defines = f"-DCXI_BACKEND -DFINE_GRAINED_TEST {device_define}"
            compile_test(args.file, output, defines, modules, args.S)

        if not args.H:
            output = f"{base_output}_hip"
            defines = f"-DHIP_BACKEND {device_define}"
            compile_test(args.file, output, defines, modules, args.S)

        if not args.C:
            output = f"{base_output}_cuda"
            defines = f"-DCUDA_BACKEND {device_define}"
            compile_test(args.file, output, defines, modules, args.S)

        if not args.T:
            output = f"{base_output}_thread"
            defines = f"-DTHREAD_BACKEND {device_define}"
            compile_test(args.file, output, defines, modules, args.S)
    else:
        compile_test(args.file, f"{base_output}", f"{device_define}", modules, "")


if __name__ == "__main__":
    main()