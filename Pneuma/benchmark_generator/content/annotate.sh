if [ "$#" -ne 1 ]; then
    echo "Usage: ./annotate.sh <dataset>"
    exit
fi
project_dir="$(dirname "$PWD")"
work_dir="$(dirname "${project_dir}")"
export PYTHONPATH=${project_dir}
python annotate_tables.py \
    --work_dir ${work_dir} \
    --dataset $1 \
    
