import os
import subprocess
import sys
import shutil

def run_command(command, cwd=None):
    """Helper to run shell commands with error handling."""
    try:
        subprocess.check_call(command, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}")
        print(f"Details: {e}")
        return False


def is_git_repo(path):
    """Checks if a directory is a valid git repository by checking for the .git folder."""
    git_folder = os.path.join(path, ".git")
    return os.path.isdir(git_folder)


def delete_specific_files(target_dir):
    """Deletes specific files or folders within the target directory."""
    # CUSTOMIZE THIS LIST
    files_to_delete = ['.gitignore', '.gitattributes', '.DS_Store', 'README.md']


    print(f"\nChecking for files to cleanup in {target_dir}...")
    for item in files_to_delete:
        item_path = os.path.join(target_dir, item)
        if os.path.exists(item_path):
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                    print(f"Deleted file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Deleted folder: {item}")
            except Exception as e:
                print(f"Failed to delete {item}. Reason: {e}")


def install_dependencies(target_dir):
    """Checks for requirements.txt and installs dependencies."""
    requirements_path = os.path.join(target_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing dependencies...")
        run_command([sys.executable, "-m", "pip", "install", "-r", requirements_path])


def install_sr_midas(repo_url, target_dir):

    print(f"\n--- Managing Installation at: {target_dir} ---")

    # 2. Check if target folder exists
    if os.path.exists(target_dir):
        # 3. Check if the target folder is actually a git repo before pulling
        if is_git_repo(target_dir):
            print(f"Detected existing Git repository.")
            choice = input(f"Update to latest version? (y/n): ").strip().lower()
            
            # If the user wants to update the repository, pull the latest changes
            if choice == 'y':
                print("Pulling latest changes...")

                # Using cwd argument is safer than -C flag for subprocess.check_call
                success = run_command(["git", "pull"], cwd=target_dir)
                
                if success:
                    print("Update successful.")
                    install_dependencies(target_dir)
                    delete_specific_files(target_dir)
                else:
                    print("Git pull failed. You may have local changes or permission issues.")
        else:
            # Folder exists, but it's not a git repo
            print(f"WARNING: The folder '{target_dir}' exists but is NOT a git repository.")
            print("Cannot run 'git pull'. Performing a fresh clone instead.")

            try:
                run_command(["git", "clone", repo_url, target_dir])
                install_dependencies(target_dir)
                delete_specific_files(target_dir)
            except Exception as e:
                print(f"Could not delete folder: {e}")
    
    # 4. If folder does not exist, Clone it
    else:
        print(f"\nERROR:The target folder '{target_dir}' does not exist.")
        print("\tThis directory was supposed to be created during MIDAS installation.")
        print("\tIf the 'MIDAS/FF_HEDM/v7' path has changed to something else in your MIDAS version, please update the 'target_dir' path in the super_res_install.py file.")
        print("\nExiting...\n")
        sys.exit(1)

if __name__ == "__main__":
    # 1. Configuring the installation directory
    current_dir = os.getcwd()
    target_dir = os.path.join(current_dir, "FF_HEDM", "v7")
    
    # SR-MIDAS github repository link
    repo_url = "https://github.com/AISDC/SR-MIDAS.git"

    install_sr_midas(repo_url, target_dir)