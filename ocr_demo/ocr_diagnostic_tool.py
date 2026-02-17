#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR诊断工具
用于分析OCR识别问题，提供详细的诊断信息
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
import re
import json

class OCRDiagnosticTool:
    """
    OCR诊断工具类
    """
    
    def __init__(self):
        """
        初始化诊断工具
        """
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
        
        # 坐标数据
        self.coordinates = {
            '_part_2': [('体重(kg)', [78, 109, 211, 189])],
            '_part_3': [('体脂率(%)', [73, 56, 210, 125])],
            '_part_4': [('骨骼肌(kg)', [75, 48, 245, 134])],
            '_part_5': [
                ('水分含量(kg)', [346, 185, 421, 243]),
                ('蛋白质(kg)', [270, 549, 524, 628]),
                ('无机盐(kg)', [268, 304, 530, 376])
            ],
            '_part_7': [('浮肿评估', [333, 44, 431, 86])],
            '_part_11': [('BMI', [327, 53, 438, 106])],
            '_part_13': [
                ('节段肌肉(kg)-左臂', [515, 239, 689, 301]),
                ('节段肌肉(kg)-右臂', [51, 244, 205, 293]),
                ('节段肌肉(kg)-躯干', [211, 696, 542, 814]),
                ('节段肌肉(kg)-左腿', [540, 558, 695, 592]),
                ('节段肌肉(kg)-右腿', [60, 558, 205, 608])
            ],
            '_part_15': [('腰臀比', [333, 56, 431, 110])],
            '_part_16': [('内脏脂肪等级', [334, 56, 421, 105])],
            '_part_17': [('基础代谢(kcal/day)', [276, 59, 404, 95])],
            '_part_18': [('健康分数', [337, 60, 429, 100])]
        }
    
    def analyze_roi_image(self, roi_path, feature_name):
        """
        分析单个ROI图像
        
        参数:
        roi_path: ROI图像路径
        feature_name: 特征名称
        
        返回:
        分析结果字典
        """
        print(f"\n=== 分析 {feature_name} ===")
        print(f"ROI路径: {roi_path}")
        
        result = {
            'feature_name': feature_name,
            'roi_path': roi_path,
            'exists': os.path.exists(roi_path),
            'file_size': 0,
            'image_size': None,
            'image_stats': {},
            'ocr_results': [],
            'diagnosis': []
        }
        
        if not result['exists']:
            result['diagnosis'].append("ROI文件不存在")
            return result
        
        try:
            # 文件大小
            result['file_size'] = os.path.getsize(roi_path)
            print(f"文件大小: {result['file_size']} bytes")
            
            # 图像信息
            img = Image.open(roi_path)
            result['image_size'] = img.size
            print(f"图像尺寸: {img.size}")
            
            # 图像统计信息
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            result['image_stats'] = {
                'mean': float(np.mean(img_gray)),
                'std': float(np.std(img_gray)),
                'min': int(np.min(img_gray)),
                'max': int(np.max(img_gray))
            }
            
            print(f"图像统计: 均值={result['image_stats']['mean']:.1f}, "
                  f"标准差={result['image_stats']['std']:.1f}, "
                  f"范围=[{result['image_stats']['min']}, {result['image_stats']['max']}]")
            
            # 图像质量诊断
            if result['image_stats']['std'] < 10:
                result['diagnosis'].append("图像对比度过低")
            if result['image_stats']['mean'] < 50 or result['image_stats']['mean'] > 200:
                result['diagnosis'].append("图像亮度异常")
            if img.size[0] < 20 or img.size[1] < 20:
                result['diagnosis'].append("ROI尺寸过小")
            
            # OCR识别测试
            ocr_result = self.ocr.ocr(roi_path, cls=False)
            result['ocr_results'] = self._parse_ocr_result(ocr_result)
            
            print(f"OCR原始结果: {ocr_result}")
            print(f"解析后结果: {result['ocr_results']}")
            
            if not result['ocr_results']:
                result['diagnosis'].append("OCR未识别到任何文本")
            else:
                # 检查是否包含数字
                has_digits = any(re.search(r'\d', item['text']) for item in result['ocr_results'])
                if not has_digits:
                    result['diagnosis'].append("识别的文本中不包含数字")
                
                # 检查置信度
                low_confidence = [item for item in result['ocr_results'] if item['confidence'] < 0.5]
                if low_confidence:
                    result['diagnosis'].append(f"部分识别结果置信度过低: {len(low_confidence)}/{len(result['ocr_results'])}")
            
        except Exception as e:
            result['diagnosis'].append(f"分析过程出错: {str(e)}")
            print(f"错误: {str(e)}")
        
        return result
    
    def _parse_ocr_result(self, ocr_result):
        """
        解析OCR结果
        """
        parsed_results = []
        
        try:
            if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                if isinstance(ocr_result[0], list):
                    for line in ocr_result[0]:
                        if isinstance(line, list) and len(line) >= 2:
                            if isinstance(line[1], list) and len(line[1]) >= 2:
                                text = str(line[1][0])
                                confidence = float(line[1][1])
                                parsed_results.append({
                                    'text': text,
                                    'confidence': confidence,
                                    'bbox': line[0] if len(line) > 0 else None
                                })
        except Exception as e:
            print(f"解析OCR结果时出错: {str(e)}")
        
        return parsed_results
    
    def analyze_image_coordinates(self, image_path, img_suffix):
        """
        分析图像和坐标的匹配度
        
        参数:
        image_path: 图像路径
        img_suffix: 图像后缀
        
        返回:
        坐标分析结果
        """
        print(f"\n=== 分析图像坐标匹配度 ===")
        print(f"图像: {image_path}")
        print(f"后缀: {img_suffix}")
        
        result = {
            'image_path': image_path,
            'img_suffix': img_suffix,
            'image_size': None,
            'coordinates': [],
            'coordinate_analysis': []
        }
        
        try:
            # 获取图像尺寸
            img = Image.open(image_path)
            result['image_size'] = img.size
            print(f"图像尺寸: {img.size}")
            
            # 获取对应的坐标
            coords_data = self.coordinates.get(img_suffix, [])
            result['coordinates'] = coords_data
            
            # 分析每个坐标
            for feature_name, coords in coords_data:
                x1, y1, x2, y2 = coords
                
                analysis = {
                    'feature_name': feature_name,
                    'coordinates': coords,
                    'valid': True,
                    'issues': []
                }
                
                # 检查坐标是否在图像范围内
                if x1 < 0 or y1 < 0 or x2 > img.size[0] or y2 > img.size[1]:
                    analysis['valid'] = False
                    analysis['issues'].append("坐标超出图像范围")
                
                # 检查坐标逻辑
                if x1 >= x2 or y1 >= y2:
                    analysis['valid'] = False
                    analysis['issues'].append("坐标逻辑错误")
                
                # 检查ROI尺寸
                roi_width = x2 - x1
                roi_height = y2 - y1
                if roi_width < 10 or roi_height < 10:
                    analysis['valid'] = False
                    analysis['issues'].append("ROI尺寸过小")
                
                analysis['roi_size'] = (roi_width, roi_height)
                
                print(f"  {feature_name}: {coords} -> ROI尺寸: {roi_width}x{roi_height}, "
                      f"有效: {analysis['valid']}")
                
                if analysis['issues']:
                    print(f"    问题: {', '.join(analysis['issues'])}")
                
                result['coordinate_analysis'].append(analysis)
                
        except Exception as e:
            print(f"坐标分析出错: {str(e)}")
        
        return result
    
    def create_visual_diagnostic(self, image_path, img_suffix, output_dir):
        """
        创建可视化诊断图像
        
        参数:
        image_path: 原始图像路径
        img_suffix: 图像后缀
        output_dir: 输出目录
        """
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/simfang.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 绘制所有坐标框
            coords_data = self.coordinates.get(img_suffix, [])
            colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
            
            for i, (feature_name, coords) in enumerate(coords_data):
                x1, y1, x2, y2 = coords
                color = colors[i % len(colors)]
                
                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 绘制标签
                draw.text((x1, y1-25), feature_name, fill=color, font=font)
            
            # 保存诊断图像
            output_path = os.path.join(output_dir, f"diagnostic_{os.path.basename(image_path)}")
            img.save(output_path)
            print(f"诊断图像已保存: {output_path}")
            
        except Exception as e:
            print(f"创建可视化诊断图像失败: {str(e)}")
    
    def diagnose_directory(self, image_dir, roi_dir=None, output_dir=None):
        """
        诊断整个目录
        
        参数:
        image_dir: 图像目录
        roi_dir: ROI目录
        output_dir: 输出目录
        """
        print(f"\n{'='*60}")
        print(f"开始诊断目录: {image_dir}")
        print(f"{'='*60}")
        
        # 创建输出目录
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        diagnostic_results = {
            'image_dir': image_dir,
            'roi_dir': roi_dir,
            'images_analyzed': 0,
            'roi_analyzed': 0,
            'issues_found': [],
            'recommendations': []
        }
        
        # 查找所有图像文件
        image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 分析每个图像
        for image_path in image_files[:3]:  # 限制分析前3个图像
            print(f"\n{'-'*40}")
            print(f"分析图像: {os.path.basename(image_path)}")
            
            # 确定图像后缀
            base_name = os.path.basename(image_path)
            img_suffix = None
            for suffix in self.coordinates.keys():
                if suffix in base_name:
                    img_suffix = suffix
                    break
            
            if not img_suffix:
                print(f"无法确定图像后缀: {base_name}")
                continue
            
            # 分析坐标匹配度
            coord_analysis = self.analyze_image_coordinates(image_path, img_suffix)
            diagnostic_results['images_analyzed'] += 1
            
            # 创建可视化诊断图像
            if output_dir:
                self.create_visual_diagnostic(image_path, img_suffix, output_dir)
            
            # 分析对应的ROI图像
            if roi_dir:
                for feature_name, coords in self.coordinates.get(img_suffix, []):
                    # 构建ROI文件路径
                    roi_filename = f"{os.path.splitext(base_name)[0]}_{feature_name.replace('(','').replace(')','').replace('/','_')}.jpg"
                    roi_path = os.path.join(roi_dir, roi_filename)
                    
                    if os.path.exists(roi_path):
                        roi_analysis = self.analyze_roi_image(roi_path, feature_name)
                        diagnostic_results['roi_analyzed'] += 1
                        
                        # 收集问题
                        if roi_analysis['diagnosis']:
                            diagnostic_results['issues_found'].extend(roi_analysis['diagnosis'])
        
        # 生成建议
        self._generate_recommendations(diagnostic_results)
        
        # 保存诊断报告
        if output_dir:
            report_path = os.path.join(output_dir, 'diagnostic_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_results, f, ensure_ascii=False, indent=2)
            print(f"\n诊断报告已保存: {report_path}")
        
        return diagnostic_results
    
    def _generate_recommendations(self, results):
        """
        生成优化建议
        """
        print(f"\n{'='*60}")
        print("诊断总结和建议")
        print(f"{'='*60}")
        
        print(f"分析了 {results['images_analyzed']} 个图像")
        print(f"分析了 {results['roi_analyzed']} 个ROI")
        print(f"发现 {len(results['issues_found'])} 个问题")
        
        # 统计问题
        issue_counts = {}
        for issue in results['issues_found']:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        print("\n主要问题:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue}: {count} 次")
        
        # 生成建议
        recommendations = []
        
        if "图像对比度过低" in issue_counts:
            recommendations.append("建议增强图像预处理，使用CLAHE或直方图均衡化")
        
        if "ROI尺寸过小" in issue_counts:
            recommendations.append("建议调整坐标配置，增大ROI区域")
        
        if "坐标超出图像范围" in issue_counts:
            recommendations.append("建议使用坐标自动调整功能适配不同分辨率")
        
        if "OCR未识别到任何文本" in issue_counts:
            recommendations.append("建议检查图像质量和预处理方案")
        
        if "识别的文本中不包含数字" in issue_counts:
            recommendations.append("建议优化数字识别的预处理流程")
        
        if not recommendations:
            recommendations.append("系统运行正常，建议监控识别准确率")
        
        results['recommendations'] = recommendations
        
        print("\n优化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR诊断工具')
    parser.add_argument('image_dir', help='图像目录')
    parser.add_argument('--roi_dir', help='ROI图像目录')
    parser.add_argument('--output_dir', default='diagnostic_output', help='诊断输出目录')
    
    args = parser.parse_args()
    
    # 创建诊断工具
    diagnostic = OCRDiagnosticTool()
    
    # 执行诊断
    results = diagnostic.diagnose_directory(
        image_dir=args.image_dir,
        roi_dir=args.roi_dir,
        output_dir=args.output_dir
    )
    
    print(f"\n诊断完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 