#!/bin/bash

go = true
while $go; do
	make maker && ./maker
	make std
	make std1

	for i in `ls ./data/`; do
		./std < ./data/$i >std.out
		./std1 < ./data/$i >std1.out
		if diff -w -b std.out std1.out; then
			echo $i WA
			go = false
			break
		fi
	done
done
