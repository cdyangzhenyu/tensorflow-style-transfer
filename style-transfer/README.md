## 环境依赖
- `gpu 1080ti`
- `cuda 10.0`
- `nvidia driver 410.104`
- `python3.7`
- `tensorflow-gpu-1.15.0`
- `numpy-1.19.1`

## 训练

```
python run_train.py --style style/wave.jpg --output model --trainDB ../../dataset/train2014 --vgg_model ../../ --test content/female_knight.jpg
```

- 训练大概40000次之后，模型效果较好，默认每1000次迭代保存一次模型。

## 冻结

- 为了使用openvino，需要先将模型冻结输出pb文件，由于该训练代码默认输入为(4, 256, 256, 3)，需要修改batch size参数，只需要在新的模型基础上重新执行以下命令即可。

```
python run_train.py --style style/wave.jpg --output model --trainDB ../../dataset/train2014 --vgg_model ../../  --test content/female_knight.jpg --checkpoint-interval 10 --batch_size 1
```

- 生成pb文件

```
python freeze_graph.py --input_meta_graph=model/final.ckpt-40010.meta --input_checkpoint=model/final.ckpt-40010 --output_graph=model/wave-40010.pb --output_node_names=output --input_binary=True
```

## 预测

- 使用ckpt文件进行预测

```
python run_test.py --style_model model/final.ckpt-40000 --content content/female_knight.jpg --output test.jpg
```

- 使用pb文件进行预测

```
python run_test_pb.py --style_model model/wave-40000.pb --content content/female_knight.jpg --output test.jpg
```

- 注意ckpt文件的分辨率更高，因为pb文件已经固定了输入大小，这里分辨率降低了很多。而ckpt模型没有固话，可以是任意输入文件。