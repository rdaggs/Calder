# new venv 
rm -rf calder_venv
python3 -m venv calder_venv
source calder_venv/bin/activate

#setup tools 
python -m pip install --upgrade pip setuptools wheel
python -m pip install ipykernel torch diffusers matplotlib numpy Pillow transformers


python -m ipykernel install --user --name calder_venv --display-name "Python (calder_venv)"
jupyter kernelspec list -- calder_venv