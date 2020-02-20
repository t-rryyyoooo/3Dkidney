#!/bin/bash

#Input
readonly TRAINING="$HOME/Desktop/data/textList/training_"
readonly VALIDATION="$HOME/Desktop/data/textList/validation_"
readonly WEIGHT="$HOME/Desktop/data/modelweight/best_"

echo -n Suffix:
read suffix
echo -n "Is the weight file's suffix the same as above?[yes/no]:"
read choice

training="${TRAINING}${suffix}.txt"
validation="${VALIDATION}${suffix}.txt"

if [ $choice = "yes" ]; then
	weight="${WEIGHT}${suffix}.hdf5"
else
	echo -n suffix:
	read newSuffix

	weight="${WEIGHT}${newSuffix}.hdf5"
fi

echo -n GPU_ID:
read id

echo $training
echo $validation
echo $weight
echo $histories

python3 rebuild.py ${training} ${weight} -t ${validation} -b 10 -e 40 -g $id
