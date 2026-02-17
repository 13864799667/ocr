#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
坐标自动调整工具
根据图片实际尺寸自动调整OCR识别坐标，解决不同分辨率图片的适配问题
"""

import os
import cv2
import numpy as np
from PIL import Image
import json
import logging

logger = logging.getLogger(__name__)

class CoordinateAdjuster:
    """
    坐标自动调整器
    """
    
    def __init__(self, reference_width=750, reference_height=1334):
        """
        初始化坐标调整器
        
        参数:
        reference_width: 参考图片宽度
        reference_height: 参考图片高度
        """
        self.reference_width = reference_width
        self.reference_height = reference_height
        
        # 基准坐标数据
        self.base_coordinates = {
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
    
    def detect_image_dimensions(self, image_path):
        """
        检测图片尺寸
        
        参数:
        image_path: 图片路径
        
        返回:
        (width, height) 元组
        """
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception as e:
            logger.error(f"无法读取图片 {image_path}: {str(e)}")
            return None, None
    
    def calculate_scale_factors(self, actual_width, actual_height):
        """
        计算缩放因子
        
        参数:
        actual_width: 实际图片宽度
        actual_height: 实际图片高度
        
        返回:
        (scale_x, scale_y) 元组
        """
        scale_x = actual_width / self.reference_width
        scale_y = actual_height / self.reference_height
        
        logger.debug(f"缩放因子: x={scale_x:.3f}, y={scale_y:.3f}")
        return scale_x, scale_y
    
    def adjust_coordinates(self, coordinates, scale_x, scale_y):
        """
        调整坐标
        
        参数:
        coordinates: 原始坐标 [x1, y1, x2, y2]
        scale_x: X轴缩放因子
        scale_y: Y轴缩放因子
        
        返回:
        调整后的坐标
        """
        x1, y1, x2, y2 = coordinates
        
        # 应用缩放因子
        new_x1 = int(x1 * scale_x)
        new_y1 = int(y1 * scale_y)
        new_x2 = int(x2 * scale_x)
        new_y2 = int(y2 * scale_y)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def get_adaptive_coordinates(self, image_path, img_suffix):
        """
        获取自适应坐标
        
        参数:
        image_path: 图片路径
        img_suffix: 图片后缀 (如 '_part_2')
        
        返回:
        调整后的坐标列表
        """
        # 检测图片尺寸
        width, height = self.detect_image_dimensions(image_path)
        if width is None or height is None:
            logger.warning(f"无法检测图片尺寸，使用原始坐标: {image_path}")
            return self.base_coordinates.get(img_suffix, [])
        
        # 计算缩放因子
        scale_x, scale_y = self.calculate_scale_factors(width, height)
        
        # 获取基准坐标
        base_coords = self.base_coordinates.get(img_suffix, [])
        
        # 调整坐标
        adjusted_coords = []
        for feature_name, coords in base_coords:
            adjusted = self.adjust_coordinates(coords, scale_x, scale_y)
            adjusted_coords.append((feature_name, adjusted))
            logger.debug(f"{feature_name}: {coords} -> {adjusted}")
        
        return adjusted_coords
    
    def validate_coordinates(self, image_path, coordinates):
        """
        验证坐标是否在图片范围内
        
        参数:
        image_path: 图片路径
        coordinates: 坐标列表
        
        返回:
        验证通过的坐标列表
        """
        width, height = self.detect_image_dimensions(image_path)
        if width is None or height is None:
            return coordinates
        
        valid_coords = []
        for feature_name, coords in coordinates:
            x1, y1, x2, y2 = coords
            
            # 检查坐标是否在图片范围内
            if (0 <= x1 < width and 0 <= y1 < height and 
                0 <= x2 <= width and 0 <= y2 <= height and 
                x1 < x2 and y1 < y2):
                valid_coords.append((feature_name, coords))
            else:
                logger.warning(f"坐标超出图片范围: {feature_name} {coords} (图片尺寸: {width}x{height})")
        
        return valid_coords
    
    def analyze_image_layout(self, image_path):
        """
        分析图片布局，尝试自动检测关键区域
        
        参数:
        image_path: 图片路径
        
        返回:
        检测到的区域信息
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 应用自适应阈值
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 过滤轮廓，寻找可能的文本区域
            text_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # 过滤太小或太大的区域
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 3.0:  # 合理的宽高比
                        text_regions.append((x, y, x+w, y+h, area))
            
            # 按面积排序
            text_regions.sort(key=lambda x: x[4], reverse=True)
            
            return text_regions[:20]  # 返回前20个区域
            
        except Exception as e:
            logger.error(f"分析图片布局失败: {str(e)}")
            return None
    
    def smart_coordinate_adjustment(self, image_path, img_suffix, feature_name):
        """
        智能坐标调整，结合图片分析和模板匹配
        
        参数:
        image_path: 图片路径
        img_suffix: 图片后缀
        feature_name: 特征名称
        
        返回:
        优化后的坐标
        """
        # 首先尝试基本的缩放调整
        adaptive_coords = self.get_adaptive_coordinates(image_path, img_suffix)
        
        # 找到对应特征的坐标
        target_coords = None
        for fname, coords in adaptive_coords:
            if fname == feature_name:
                target_coords = coords
                break
        
        if target_coords is None:
            return None
        
        # 分析图片布局
        regions = self.analyze_image_layout(image_path)
        if regions is None:
            return target_coords
        
        # 寻找最接近目标坐标的区域
        x1, y1, x2, y2 = target_coords
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        best_region = None
        min_distance = float('inf')
        
        for region in regions:
            rx1, ry1, rx2, ry2, _ = region
            rcenter_x = (rx1 + rx2) / 2
            rcenter_y = (ry1 + ry2) / 2
            
            # 计算距离
            distance = np.sqrt((center_x - rcenter_x)**2 + (center_y - rcenter_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                best_region = region
        
        # 如果找到了相近的区域，使用该区域的坐标
        if best_region and min_distance < 100:  # 距离阈值
            rx1, ry1, rx2, ry2, _ = best_region
            logger.debug(f"为 {feature_name} 找到更好的坐标: {[rx1, ry1, rx2, ry2]}")
            return [rx1, ry1, rx2, ry2]
        
        return target_coords
    
    def save_adjusted_coordinates(self, adjusted_coords, output_file):
        """
        保存调整后的坐标到文件
        
        参数:
        adjusted_coords: 调整后的坐标字典
        output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(adjusted_coords, f, ensure_ascii=False, indent=2)
            logger.info(f"调整后的坐标已保存到: {output_file}")
        except Exception as e:
            logger.error(f"保存坐标文件失败: {str(e)}")
    
    def load_adjusted_coordinates(self, input_file):
        """
        从文件加载调整后的坐标
        
        参数:
        input_file: 输入文件路径
        
        返回:
        坐标字典
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载坐标文件失败: {str(e)}")
            return {}

def create_coordinate_adjuster():
    """
    创建坐标调整器实例
    """
    return CoordinateAdjuster()

def test_coordinate_adjustment(image_dir):
    """
    测试坐标调整功能
    
    参数:
    image_dir: 测试图片目录
    """
    adjuster = create_coordinate_adjuster()
    
    # 查找测试图片
    test_images = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print(f"在 {image_dir} 中没有找到图片文件")
        return
    
    # 测试第一张图片
    test_image = test_images[0]
    print(f"测试图片: {test_image}")
    
    # 检测尺寸
    width, height = adjuster.detect_image_dimensions(test_image)
    print(f"图片尺寸: {width}x{height}")
    
    # 测试坐标调整
    for suffix in ['_part_2', '_part_3', '_part_4']:
        coords = adjuster.get_adaptive_coordinates(test_image, suffix)
        print(f"{suffix}: {coords}")

if __name__ == "__main__":
    # 测试代码
    test_image_dir = "all_color_output"
    if os.path.exists(test_image_dir):
        test_coordinate_adjustment(test_image_dir)
    else:
        print("测试目录不存在") 