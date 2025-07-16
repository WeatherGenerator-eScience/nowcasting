#!/bin/bash
# Script writen by Tim Vlemmix

# Create a query via https://robukis-core.widi.knmi.cloud/servlet/selection/SelectTableElementGroup/ (onderaan "Show recipe")

# Copy the input query file. This file defines the specific AWS data to be downloaded.
cp table_10_mindata_zonder_luchtvochtigheid.in table.in

# Define the output directory for downloaded CSV files
output_dir="path/to/storage/dir"

for year in {2020..2023}; do
        for month in {01..12} ; do
                # Calculate delta_year for correct year increment in ym2 for December
                delta_year=$(echo "($month/12)"|bc)
                # Format current year and month (e.g., 202001)
                ym1=$year$month
                ym2=$(echo "($year + $delta_year)"|bc)$(printf %02d `echo "($month % 12)+1"|bc`)
                echo $ym1"_"$ym2 
                # Ideally downloading would be done for one file a month, but this process sometimes comes to a halt.
                # Downloading one month in 5 parts is more reliable.
                # The commented-out line below shows the original attempt for a single monthly download.
                # wget --header=Content-Type:application/x-www-form-urlencoded --post-file=table.in -O "$output_dir/RobuKIS_table_$ym1.csv" "https://robukis-core.widi.knmi.cloud/servlet/download/table/${ym1}01_000000/${ym2}01_000000/CSV"
        
                wget --header=Content-Type:application/x-www-form-urlencoded --post-file=table.in -O $output_dir"RobuKIS_table_${ym1}01_000000_${ym1}08_000000.csv" "https://robukis-core.widi.knmi.cloud/servlet/download/table/${ym1}01_000000/${ym1}08_000000/CSV"
  
                wget --header=Content-Type:application/x-www-form-urlencoded --post-file=table.in -O $output_dir"RobuKIS_table_${ym1}08_000000_${ym1}16_000000.csv" "https://robukis-core.widi.knmi.cloud/servlet/download/table/${ym1}08_000000/${ym1}16_000000/CSV"

                wget --header=Content-Type:application/x-www-form-urlencoded --post-file=table.in -O $output_dir"RobuKIS_table_${ym1}16_000000_${ym1}24_000000.csv" "https://robukis-core.widi.knmi.cloud/servlet/download/table/${ym1}16_000000/${ym1}24_000000/CSV"
  
                wget --header=Content-Type:application/x-www-form-urlencoded --post-file=table.in -O $output_dir"RobuKIS_table_${ym1}24_000000_${ym2}01_000000.csv" "https://robukis-core.widi.knmi.cloud/servlet/download/table/${ym1}24_000000/${ym2}01_000000/CSV"  

        done
done