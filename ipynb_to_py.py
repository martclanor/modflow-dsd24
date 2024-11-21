import os
import subprocess


def convert_ipynb_to_py(ipynb_file):
    """Convert .ipynb file to .py using nbconvert."""
    try:
        subprocess.run(
            ["jupyter", "nbconvert", "--to", "script", ipynb_file], check=True
        )
        print(f"Converted {ipynb_file} to .py")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {ipynb_file}: {e}")


def convert_all_ipynb_to_py(root_dir):
    """Recursively convert all .ipynb files to .py"""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".ipynb"):
                ipynb_file = os.path.join(dirpath, filename)
                convert_ipynb_to_py(ipynb_file)


if __name__ == "__main__":
    root_directory = "."
    convert_all_ipynb_to_py(root_directory)
