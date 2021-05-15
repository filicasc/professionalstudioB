#!/bin/bash

echo "date,open,high,low,close,volume,adj_close,Name" > SP500_latest_stock.csv
cd individual_stocks_5yr
files=$(ls *.csv)
for file in $files
do
	tail -n +2 $file >> ../SP500_latest_stock.csv
done