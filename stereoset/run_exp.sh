export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
#python experiment.py --persistent_dir "/export/home/kraft/data/stereoset_data" --model "BertForMaskedLM" --model_name_or_path "bert-base-uncased"
#python experiment.py --persistent_dir "/export/home/kraft/data/stereoset_data" --model "ErnieForMaskedLM" --model_name_or_path "nghuyong/ernie-2.0-base-enqq"

python stereoset_evaluation.py --persistent_dir "/export/home/kraft/data/stereoset_data" --predictions_file "/export/home/kraft/data/stereoset_data/results/stereoset/stereoset_m-ErnieForMaskedLM_c-nghuyong-ernie-2.0-base-en.json"  --output_file "/export/home/kraft/data/stereoset_data/results/stereoset/results.json"
#python stereoset_evaluation.py --persistent_dir "/export/home/kraft/data/stereoset_data" --predictions_file "/export/home/kraft/data/stereoset_data/results/stereoset/stereoset_m-BertForMaskedLM_c-bert-large-uncased.json"  --output_file "/export/home/kraft/data/stereoset_data/results/stereoset/results.json"