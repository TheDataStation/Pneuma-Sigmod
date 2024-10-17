if [ "$#" -ne 3 ]; then
    echo "Usage: ./import_tables.sh <dataset> <file_name_caption> <truncate>"
    exit
fi
project_dir="$(dirname "$PWD")"
work_dir="$(dirname "${project_dir}")"
python table_from_csv.py \
    --work_dir ${work_dir} \
    --dataset $1 \
    --file_name_title $2 \
    --truncate $3 \