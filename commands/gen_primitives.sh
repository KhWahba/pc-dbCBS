cd ../build
DYN=${1:-"mujocoquad_empty"}
PRIM=${2:-"200"}
OUT_FILE=${3:-"../motion_primitives/mujoco_quad3d/quad3d.bin"}
VIZ_FILE=${4:-"../motion_primitives/mujoco_quad3d/quad3d_viz.html"}

./deps/dynoplan/main_primitives --mode_gen_id 0 --dynamics $DYN --models_base_path ../deps/dynoplan/dynobench/models/ --max_num_primitives $PRIM --out_file $OUT_FILE --solver_id 0 --cfg_file ../configs/opt.yaml 

./deps/dynoplan/main_primitives --mode_gen_id 1 --dynamics $DYN --models_base_path ../deps/dynoplan/dynobench/models/ --max_num_primitives $PRIM --in_file $OUT_FILE --solver_id 0 --cfg_file ../configs/opt.yaml 

./deps/dynoplan/main_primitives --mode_gen_id 2 --in_file  ${OUT_FILE}.im.bin    --max_num_primitives -1  --max_splits 4  --max_length_cut 40  --min_length_cut 20 --dynamics $DYN --models_base_path ../deps/dynoplan/dynobench/models/

python3 ../scripts/visualize_prims.py --prims ${OUT_FILE}.im.bin.sp.bin.yaml --output ${VIZ_FILE}_splitted.html --num_samples 20
python3 ../scripts/visualize_prims.py --prims ${OUT_FILE}.im.bin.yaml --output ${VIZ_FILE}.html --num_samples 20
