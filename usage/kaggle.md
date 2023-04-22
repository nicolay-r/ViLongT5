### Kaggle Setup 

First, it is necessary to setup `python-3.8+`. We have experiment with the version 3.9.

```
sudo apt-get install python3.9
sudo apt install python3.9-distutils

# Installing pip.
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.9 get-pip.py

# Done!
python3.9 -m pip  ...
```

Setup `venv` which allows us to adopt `python` of the 3.9 version:
```
sudo apt-get install python3.9-venv
python3.9 -m venv ./venv
source venv/bin/activate
```
