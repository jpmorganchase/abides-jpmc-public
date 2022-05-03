python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
cd abides-core
python setup.py develop
cd ../abides-markets
python setup.py develop
cd ..