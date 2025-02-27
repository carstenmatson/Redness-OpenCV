import os

def generate_tree(directory, prefix=""):
    """Generate a directory tree."""
    files = os.listdir(directory)
    for i, file in enumerate(files):
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            print(f"{prefix}├── {file}/")
            generate_tree(path, prefix + "│   ")
        else:
            if i == len(files) - 1:
                print(f"{prefix}└── {file}")
            else:
                print(f"{prefix}├── {file}")

if __name__ == "__main__":
    project_dir = "."  # Current directory (root of the project)
    generate_tree(project_dir)