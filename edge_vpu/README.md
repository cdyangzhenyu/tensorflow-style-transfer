## 依赖
- `python3.7.6`
- `openvino_2020.1.023`
- `opencv-python-3.4.2.17`
- `NCS2`
- `Ubuntu18.04`

## 生成openvino需要的模型文件

- 将train之后得到的pb文件进行转换

`python /opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer/mo_tf.py --input_model ../models/tangyan.pb --output_dir tangyan/`

## 使用openvino进行推理（VPU/CPU）

- 单线程推理，不加`-d`默认使用cpu：

`python camera_vpu_demo.py -m lrmodels/tangyan/tangyan.xml -d MYRIAD`

- 多线程多vpu推理：

`python async_multi_vpu.py`
