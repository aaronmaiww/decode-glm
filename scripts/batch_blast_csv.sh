#!/bin/bash
# Query sequences against recreated batch databases
# Usage: ./script.sh <query_file_path>

# Check if query file is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <query_file_path>"
    echo "Example: $0 data/query_sequences/CMV_enhancer.fasta"
    exit 1
fi

query_file="$1"
mkdir -p blast_results

# Check if query file exists
if [ ! -f "$query_file" ]; then
    echo "Error: Query file $query_file not found!"
    exit 1
fi

# Extract query filename for results naming
query_basename=$(basename "$query_file" .fasta)
results_dir="blast_results_${query_basename}"
mkdir -p "$results_dir"

batch_size=100
batch_num=1

total_files=$(find . -name "*.fna" | wc -l)
echo "Total files: $total_files"
echo "Query file: $query_file"
echo "Results directory: $results_dir"
echo "Starting batch processing..."

# Process in batches - recreate each database temporarily
for ((start=1; start<=total_files; start+=batch_size)); do
    end=$((start + batch_size - 1))
    
    echo "Processing batch $batch_num (files $start-$end)"
    
    # Get batch files
    find . -name "*.fna" | sed -n "${start},${end}p" > batch${batch_num}_files.txt
    
    # Create database
    echo "  Creating database for batch $batch_num..."
    cat $(cat batch${batch_num}_files.txt) | makeblastdb -in - -dbtype nucl -out batch${batch_num}_db -title "Batch $batch_num Database"
    
    # Create CSV with header
    echo "query_id,subject_id,percent_identity,alignment_length,mismatches,gap_opens,query_start,query_end,subject_start,subject_end,evalue,bitscore" > "${results_dir}/${query_basename}_vs_batch${batch_num}.csv"
    
    # Run BLAST with CSV output
    echo "  Running BLAST for batch $batch_num..."
    blastn -query "$query_file" -db batch${batch_num}_db -outfmt "10 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" >> "${results_dir}/${query_basename}_vs_batch${batch_num}.csv"
    
    # Clean up intermediate files
    rm batch${batch_num}_files.txt
    rm batch${batch_num}_db.*
    
    echo "  Completed batch $batch_num"
    ((batch_num++))
done

echo "All batches complete!"

# Combine all results
echo "Combining results..."
if ls "${results_dir}/${query_basename}_vs_batch"*.csv 1> /dev/null 2>&1; then
    # Take header from first file
    head -1 "${results_dir}/${query_basename}_vs_batch1.csv" > "${results_dir}/${query_basename}_all_results_combined.csv"
    
    # Add all data rows (skip headers from each file)
    for file in "${results_dir}/${query_basename}_vs_batch"*.csv; do
        tail -n +2 "$file" >> "${results_dir}/${query_basename}_all_results_combined.csv"
    done
    
    total_hits=$(tail -n +2 "${results_dir}/${query_basename}_all_results_combined.csv" | wc -l)
    echo "Combined results saved as: ${results_dir}/${query_basename}_all_results_combined.csv"
    echo "Total BLAST hits found: $total_hits"
else
    echo "No result files found to combine."
fi
