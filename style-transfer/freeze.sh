python freeze_graph.py \
	--input_meta_graph=./model/test.ckpt.meta \
	--input_checkpoint=./model/test.ckpt \
	--output_graph=./model/test.pb \
	--output_node_names=output \
	--input_binary=True