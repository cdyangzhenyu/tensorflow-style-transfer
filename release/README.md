## 运行环境依赖
- `python3.7`
- `tensorflow-1.15`
- `opencv-python-3.4.2`
- `cuda-10`
- `Ubuntu18.04`
- `GPU-1080`

## 安装部署发布版本

### 编译
- 修改src下面代码后重新编译生成bin文件
`sh gen_bin.sh`

### 打包
- 确保需要的模型已经在models目录下
- 执行以下命令生成tar包`style_transfer.tar.gz`
- `sh gen_tar.sh`

### 安装及软件启动
- 将上面打好的tar包拷贝到目标机器对应目录里
```
tar -zxvf style_transfer.tar.gz
cd style_transfer
chmod +x *.sh
./install.sh
```

### Ubuntu18开机自动启动
- 搜索`Startup Applications`应用
- 添加新的项Command为`/root/startup.sh`
