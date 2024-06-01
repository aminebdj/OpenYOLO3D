export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=750
CURR_QUERY=150

python predict.py \
general.experiment_name="arkitscenes" \
general.project_name="arktiscenes" \
general.checkpoint="checkpoints/scannet200/scannet200_benchmark.ckpt" \
data/datasets=scannet200 \
general.num_targets=201 \
data.num_labels=200 \
general.eval_on_segments=false \
general.train_on_segments=false \
general.train_mode=false \
model.num_queries=${CURR_QUERY} \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN} \
general.export=true \
data.test_mode=test \
general.export_threshold=${CURR_T}