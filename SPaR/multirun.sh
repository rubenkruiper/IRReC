#!/bin/sh
DIR='experiments/'
# loop over files in folder experiments
for FILE in $(ls "$DIR" | grep .json); do
	filename=$(basename "$FILE")
	echo "Will run experiment defined in: $filename"
	# if not simple_tagger then run and wait
	if [[ "$filename" == *"simplest_tagger"* ]]; then
		wait # do nothing
	if [[ "$filename" == *"attention_tagger"* ]]; then
		wait # do nothing
	else
		# run and store model on HDD
		fname="${filename%.*}"
		python run_tagger.py -c "$DIR$filename" -m "/mnt/sda/mwe/$fname"
		wait
		mv $DIR$filename $DIR"finished/"$filename
	fi
done