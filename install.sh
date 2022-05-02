python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
cd abides-core
python3 setup.py install
cd ../abides-markets
python3 setup.py install
cd ../abides-gym
python3 setup.py install
cd ..
