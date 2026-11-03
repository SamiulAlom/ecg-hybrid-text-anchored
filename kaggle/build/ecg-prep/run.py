import glob, os, sys, subprocess
CODE = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
sys.path.insert(0, CODE)
print("code at", CODE)

subprocess.run([sys.executable, os.path.join(CODE, "preprocess.py"), "--roots", "/kaggle/input",
                "--out", "/kaggle/working/prep"], check=True)
