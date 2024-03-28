# Python例程 <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 使用方式](#22-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号  |  Python例程       | 说明                            |
| ---- | ----------------  | ------------------------------- |
| 1    |    whisper.py     |         使用SAIL推理             |


## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```
您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，相关功能还未发布，这里暂时提供一个可用的sophon-sail版本，x86/arm PCIe环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖

#x86 pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/pcie/sophon-3.7.0-py3-none-any.whl
pip3 install sophon-3.7.0-py3-none-any.whl

#arm pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/arm_pcie/sophon_arm_pcie-3.7.0-py3-none-any.whl
pip3 install sophon_arm_pcie-3.7.0-py3-none-any.whl
```
如果您需要其他版本的sophon-sail，或者遇到glibc版本问题（pcie环境常见），可以通过以下命令下载源码，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)自己编译sophon-sail。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/sophon-sail_20240226.tar.gz
tar xvf sophon-sail_20240226.tar.gz
```
### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```
由于本例程需要的sophon-sail版本较新，这里提供一个可用的sophon-sail whl包，SoC环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/soc/sophon_arm-3.7.0-py3-none-any.whl #arm soc, py38
```
如果您需要其他版本的sophon-sail，可以参考上一小节，下载源码自己编译。

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: whisper.py wavfile/path [--model MODEL][--bmodel_dir BMODEL_DIR] [--dev_id DEV_ID] [--output_dir OUTPUT_DIR] [--output_format OUTPUT_FORMAT]
--model: 选择模型尺寸 small/base/medium
--bmodel_dir: 用于推理的bmodel文件夹路径；
--dev_id: 用于推理的tpu设备id，默认为0；
--output_dir：模型输出的存放路径；
--output_format: 模型输出的保存格式；
--help: 输出帮助信息
```

### 2.2 使用方式
测试单个语音文件
```bash
export PATH=$PATH:/opt/sophon/sophon-ffmpeg-latest/bin

python3 whisper.py ../datasets/test/demo.wav --model base --bmodel_dir ../models/BM1684X --dev_id 0  --output_dir ./result/ --output_format txt
```

测试语音数据集

```bash
export PATH=$PATH:/opt/sophon/sophon-ffmpeg-latest/bin

python3 whisper.py ../datasets/aishell_S0764/ --model base --bmodel_dir ../models/BM1684X --dev_id 0  --output_dir ./result/ --output_format txt
```


