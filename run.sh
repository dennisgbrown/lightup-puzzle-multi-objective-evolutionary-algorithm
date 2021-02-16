#!/bin/bash

# Call start.py using python 3.6 on the campus Linux machines 
# with a backup standard Python3 call for my convenience
/linux_apps/python-3.6.1/bin/python3 ./code/start.py $1 $2 || python3 ./code/start.py $1 $2
