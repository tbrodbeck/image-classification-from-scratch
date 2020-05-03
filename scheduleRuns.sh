#! /bin/bash

for RUNNR in 1 2 3 .. N
do
	for SETUPNR in 1 2 3 .. N
    do
        python searchBestSetup.py $SETUPNR $RUNNR
    done
done