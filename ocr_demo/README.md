# OCR Demo

基于PaddleOCR的图像文字识别演示项目，支持长图分割、批量OCR识别及结果导出到Excel。

## 功能特点

- 支持长图自动分割处理（按固定高度或黑色分隔线）
- 基于PaddleOCR的高精度OCR文字识别
- 支持批量处理图片文件夹
- 支持识别结果导出到Excel表格
- 支持保存ROI(感兴趣区域)图像用于结果验证
- 提供预处理功能以提高OCR识别率

## 环境要求

项目依赖以下库，可通过`requirements.txt`安装：

```
paddlepaddle==2.4.2
paddleocr>=2.6.0.3
PyQt5==5.15.4
openpyxl==3.0.9
pandas==1.3.4
Pillow==9.0.0
numpy==1.21.4
opencv-python==4.7.0.72
pytesseract==0.3.10
xlrd==2.0.1
xlwt==1.3.0
pyinstaller==4.9
```

安装命令：
```
pip install -r requirements.txt
```

## 使用方法

### 一键处理（分割+OCR识别）

使用`one_key_process.py`脚本可以一键完成图片分割和OCR识别：

```
python one_key_process.py <输入图片目录> [--output_dir <输出目录>] [--excel <Excel输出文件>] 
                         [--roi_dir <ROI保存目录>] [--split_mode <分割模式>] [--no_split]
```

示例：
```
python one_key_process.py datasets/ --output_dir all_output --excel results.xlsx --roi_dir roi_images
```

### 仅执行图片分割

使用`split_image.py`脚本可以将长图分割成多个小图：

```
python split_image.py <输入图片或目录> <输出目录> [--height <分割高度>] 
                     [--by-color] [--black <黑色阈值>] [--ratio <黑色比例>] 
                     [--min-height <最小分隔线高度>] [--max-height <最大分隔线高度>]
```

分割模式：
- 按固定高度分割（默认）：指定`--height`参数设置每个分割图片的高度
- 按黑色分隔线分割：添加`--by-color`参数，并可调整黑色阈值和比例

### 批量OCR识别导出到Excel

使用`batch_ocr_to_excel.py`脚本可以批量处理图片并将OCR结果导出到Excel：

```
python batch_ocr_to_excel.py <图片目录> [坐标文件] [输出Excel文件] [--recursive] [--save-roi] [--roi-dir <ROI保存目录>]
```

示例：
```
python batch_ocr_to_excel.py min_height_test results.xlsx --recursive --save-roi --roi-dir batch_roi
```

## 参考坐标文件

项目使用坐标文件定义要识别的区域，格式为：

```
特征名称: x1, y1, x2, y2
```

其中(x1,y1)和(x2,y2)分别表示要识别区域的左上角和右下角坐标。可参考项目中的`参考坐标区域.txt`文件。

## 注意事项

1. 对于图片质量较差的情况，可以调整预处理参数以提高识别率
2. 保存的ROI图像可用于验证识别结果是否准确
3. 对于复杂表格，建议使用按黑色分隔线分割的方式
4. 对于包含多个子目录的批量处理，可使用`--recursive`参数 