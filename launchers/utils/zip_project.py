import glob
import os
import os.path as path
import subprocess
from zipfile import ZipFile

ZIP_IGNORE = [
    'data',
    'exp',
    'doc',
    'venv',
    'README'
    'setup'
]

if 'PROJECTDIR' in os.environ:
    PROJECTDIR = os.environ['PROJECTDIR']
else:
    PROJECTDIR = os.getcwd()

def zip_project(log_dir):
    zip_path = path.join(log_dir, "project_snapshot.zip")
    print('Saving project codes to {}'.format(zip_path))
    with ZipFile(zip_path, mode="x") as zipfile:
        # Git info
        git_info = subprocess.run(
            ['git', 'log', '-n 1'],
            stdout=subprocess.PIPE).stdout.decode('utf-8')
        git_info += subprocess.run(
            ['git', 'status'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        zipfile.writestr("git_info.txt", git_info)

        # Files
        for f in glob.iglob(
                path.join(PROJECTDIR, "**"), recursive=True):
            # exclude some files
            keep = True
            keep = keep and path.isfile(f)
            keep = keep and not "__pycache__" in f
            for i in ZIP_IGNORE:
                if f.startswith(path.join(PROJECTDIR, i)):
                    keep = False
                    break
            if keep:
                print("Zipping {}...".format(f))
                zipfile.write(f, arcname=path.relpath(f, PROJECTDIR))
