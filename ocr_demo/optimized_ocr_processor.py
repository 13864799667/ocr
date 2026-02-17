#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化的OCR处理器
解决图片识别问题，提高数字识别精度
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from paddleocr import PaddleOCR
import re
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedOCRProcessor:
    """
    优化的OCR处理器类
    """
    
    def __init__(self, lang='ch', use_angle_cls=True, use_gpu=True):
        """
        初始化OCR处理器
        
        参数:
        lang: 语言设置
        use_angle_cls: 是否使用角度分类
        use_gpu: 是否使用GPU
        """
        try:
            # 优化的PaddleOCR参数设置
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                # 针对数字识别的优化参数
                det_max_side_len=1280,  # 增大检测最大边长，适应高分辨率
                rec_batch_num=6,        # 优化批处理大小
                drop_score=0.3          # 降低置信度阈值，保留更多候选
            )
            logger.info("OCR处理器初始化成功")
        except Exception as e:
            logger.error(f"OCR处理器初始化失败: {str(e)}")
            raise
    
    def adaptive_preprocess_image(self, img, feature_name):
        """
        自适应图像预处理
        根据特征类型和图像特征选择最佳预处理方案
        
        参数:
        img: PIL图像对象
        feature_name: 特征名称
        
        返回:
        预处理后的图像列表（多种预处理方案）
        """
        processed_images = []
        
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 方案1: 原图（轻微增强）
        try:
            enhancer = ImageEnhance.Contrast(img)
            enhanced_img = enhancer.enhance(1.2)
            processed_images.append(("original_enhanced", enhanced_img))
        except Exception as e:
            logger.warning(f"原图增强失败: {str(e)}")
            processed_images.append(("original", img))
        
        # 方案2: 灰度+自适应阈值
        try:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            # 使用自适应阈值而不是固定OTSU
            adaptive_thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            processed_img = Image.fromarray(adaptive_thresh)
            processed_images.append(("adaptive_thresh", processed_img))
        except Exception as e:
            logger.warning(f"自适应阈值处理失败: {str(e)}")
        
        # 方案3: 针对数字优化的预处理
        if self._is_numeric_feature(feature_name):
            try:
                # 对数字特征使用特殊预处理
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # 轻微高斯模糊去噪
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                
                # 使用Otsu阈值
                _, otsu_thresh = cv2.threshold(blurred, 0, 255, 
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 轻微形态学操作
                kernel = np.ones((2, 2), np.uint8)
                morphed = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
                
                processed_img = Image.fromarray(morphed)
                # 增强对比度
                enhancer = ImageEnhance.Contrast(processed_img)
                processed_img = enhancer.enhance(1.5)
                
                processed_images.append(("numeric_optimized", processed_img))
            except Exception as e:
                logger.warning(f"数字优化处理失败: {str(e)}")
        
        # 方案4: 高对比度处理（适用于低对比度图像）
        try:
            # 检查图像对比度
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            
            if contrast < 30:  # 低对比度图像
                # 使用CLAHE增强对比度
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(gray)
                
                # 应用锐化
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                processed_img = Image.fromarray(sharpened)
                processed_images.append(("high_contrast", processed_img))
        except Exception as e:
            logger.warning(f"高对比度处理失败: {str(e)}")
        
        # 方案5: 特殊处理浮肿评估和健康分数
        if "浮肿评估" in feature_name or "健康分数" in feature_name:
            try:
                # 应用更强的对比度增强
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # 使用更强的CLAHE增强对比度
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
                enhanced = clahe.apply(gray)
                
                # 使用更强的二值化
                _, binary = cv2.threshold(enhanced, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 应用开运算去除噪点
                kernel = np.ones((2, 2), np.uint8)
                opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # 边缘增强
                edge_enhanced = cv2.Laplacian(opened, cv2.CV_8U)
                combined = cv2.addWeighted(opened, 0.7, edge_enhanced, 0.3, 0)
                
                processed_img = Image.fromarray(combined)
                processed_images.append(("special_enhanced", processed_img))
            except Exception as e:
                logger.warning(f"特殊增强处理失败: {str(e)}")
        
        return processed_images
    
    def _is_numeric_feature(self, feature_name):
        """
        判断是否为数字特征
        """
        numeric_keywords = ['kg', '%', 'BMI', '分数', '等级', 'kcal', '比']
        return any(keyword in feature_name for keyword in numeric_keywords)
    
    def extract_numbers_advanced(self, text, feature_name):
        """
        高级数字提取
        根据特征类型使用不同的提取策略
        
        参数:
        text: OCR识别的文本
        feature_name: 特征名称
        
        返回:
        提取的数字文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 清理文本
        cleaned_text = re.sub(r'[^\d\.\-\+]', ' ', text)
        
        # 根据特征类型使用不同的正则表达式
        if "%" in feature_name:
            # 百分比：寻找0-100之间的数字
            numbers = re.findall(r'\b(?:[0-9]|[1-9][0-9]|100)(?:\.[0-9]+)?\b', cleaned_text)
        elif "BMI" in feature_name:
            # BMI：通常在15-40之间
            numbers = re.findall(r'\b(?:[1-3][0-9]|[4][0]|[1][5-9])(?:\.[0-9]+)?\b', cleaned_text)
        elif "健康分数" in feature_name:
            # 健康分数：通常在0-10之间
            numbers = re.findall(r'\b(?:[0-9]|10)(?:\.[0-9]+)?\b', cleaned_text)
            # 如果没有找到数字，尝试另一种匹配方式（可能数字被识别为文本）
            if not numbers:
                # 尝试匹配常见的数字错误识别模式
                digit_map = {'o': '0', 'O': '0', 'l': '1', 'I': '1', 'z': '2', 'Z': '2', 
                            's': '5', 'S': '5', 'b': '6', 'B': '8', 'g': '9'}
                corrected_text = ''.join(digit_map.get(c, c) for c in text)
                cleaned_corrected = re.sub(r'[^\d\.\-\+]', ' ', corrected_text)
                numbers = re.findall(r'\b(?:[0-9]|10)(?:\.[0-9]+)?\b', cleaned_corrected)
        elif "浮肿评估" in feature_name:
            # 浮肿评估：通常是单个数字0-5
            numbers = re.findall(r'\b[0-5](?:\.[0-9]+)?\b', cleaned_text)
            # 如果没有找到符合范围的数字，尝试提取任何数字
            if not numbers:
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', cleaned_text)
        elif "节段肌肉" in feature_name:
            # 节段肌肉：通常在0-50之间
            numbers = re.findall(r'\b(?:[0-4]?[0-9]|50)(?:\.[0-9]+)?\b', cleaned_text)
        else:
            # 通用数字提取
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', cleaned_text)
        
        if numbers:
            # 选择最合理的数字
            return self._select_best_number(numbers, feature_name)
        
        return ""
    
    def _select_best_number(self, numbers, feature_name):
        """
        从多个识别的数字中选择最合理的一个
        """
        if not numbers:
            return ""
        
        # 转换为浮点数并过滤
        valid_numbers = []
        for num_str in numbers:
            try:
                num = float(num_str)
                # 根据特征类型进行合理性检查
                if self._is_reasonable_value(num, feature_name):
                    valid_numbers.append((num, num_str))
            except ValueError:
                continue
        
        if not valid_numbers:
            # 如果没有合理的数字，返回第一个
            return numbers[0]
        
        # 返回最合理的数字（优先选择在合理范围内的）
        return valid_numbers[0][1]
    
    def _is_reasonable_value(self, value, feature_name):
        """
        检查数值是否在合理范围内
        """
        ranges = {
            '体重': (30, 200),
            '体脂率': (5, 50),
            '骨骼肌': (15, 50),
            '水分含量': (20, 60),
            'BMI': (15, 40),
            '节段肌肉': (1, 30),
            '腰臀比': (0.7, 1.2),
            '内脏脂肪等级': (1, 20),
            '基础代谢': (800, 3000),
            '健康分数': (0, 10),
            '浮肿评估': (0, 5)
        }
        
        for key, (min_val, max_val) in ranges.items():
            if key in feature_name:
                return min_val <= value <= max_val
        
        return True  # 如果没有特定范围，认为合理
    
    def perform_ocr_with_fallback(self, image_path, coords, feature_name, 
                                save_roi=True, roi_dir="roi_images"):
        """
        执行OCR识别，使用多种预处理方案和错误处理
        
        参数:
        image_path: 图片路径
        coords: 坐标列表 [x1, y1, x2, y2]
        feature_name: 特征名称
        save_roi: 是否保存ROI图像
        roi_dir: ROI图像保存目录
        
        返回:
        识别的文本
        """
        try:
            # 打开图像
            img = Image.open(image_path)
            x1, y1, x2, y2 = coords
            
            # 验证坐标
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"无效坐标: {coords}")
                return ""
            
            # 裁剪ROI
            roi = img.crop((x1, y1, x2, y2))
            
            # 检查ROI大小
            if roi.size[0] < 10 or roi.size[1] < 10:
                logger.warning(f"ROI太小: {roi.size}")
                return ""
            
            # 保存原始ROI
            if save_roi:
                os.makedirs(roi_dir, exist_ok=True)
                base_name = os.path.basename(image_path)
                roi_name = f"{os.path.splitext(base_name)[0]}_{feature_name.replace('(','').replace(')','').replace('/','_')}.jpg"
                roi_path = os.path.join(roi_dir, roi_name)
                roi.save(roi_path)
            
            # 获取多种预处理方案
            processed_images = self.adaptive_preprocess_image(roi, feature_name)
            
            best_result = ""
            best_confidence = 0
            
            # 尝试每种预处理方案
            for method_name, processed_img in processed_images:
                try:
                    # 保存临时文件
                    temp_path = os.path.join(roi_dir, f"temp_{method_name}.jpg")
                    processed_img.save(temp_path)
                    
                    # 执行OCR
                    result = self.ocr.ocr(temp_path, cls=False)
                    
                    # 安全地提取结果
                    text, confidence = self._extract_ocr_result(result)
                    
                    if text and confidence > best_confidence:
                        best_result = text
                        best_confidence = confidence
                        logger.debug(f"方法 {method_name} 识别结果: {text} (置信度: {confidence:.3f})")
                    
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    logger.warning(f"预处理方案 {method_name} 失败: {str(e)}")
                    continue
            
            # 特殊处理浮肿评估和健康分数
            if ("浮肿评估" in feature_name or "健康分数" in feature_name) and not best_result:
                # 如果常规OCR无结果，尝试单字符识别
                try:
                    logger.info(f"尝试对 {feature_name} 使用单字符识别方法")
                    # 特殊调整参数的OCR实例
                    single_char_ocr = PaddleOCR(
                        use_angle_cls=False,
                        lang='ch',
                        rec_char_dict_path='ppocr/utils/ppocr_keys_v1.txt',
                        rec_batch_num=1,
                        rec_algorithm='CRNN',
                        det=False  # 禁用检测，直接识别
                    )
                    
                    # 增强ROI处理
                    enhanced_roi = ImageEnhance.Contrast(roi).enhance(2.0)
                    enhanced_roi = enhanced_roi.filter(ImageFilter.SHARPEN)
                    temp_path = os.path.join(roi_dir, f"temp_enhanced_single.jpg")
                    enhanced_roi.save(temp_path)
                    
                    # 执行直接识别
                    result = single_char_ocr.ocr(temp_path, cls=False, det=False)
                    if result and len(result) > 0:
                        text = result[0][0][0] if result[0] else ""
                        confidence = result[0][0][1] if result[0] else 0
                        logger.info(f"单字符识别结果: {text} (置信度: {confidence})")
                        if text:
                            best_result = text
                    
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    logger.warning(f"单字符识别失败: {str(e)}")
            
            # 提取数字
            if best_result:
                logger.info(f"特征 {feature_name} 原始识别结果: {best_result}")
                cleaned_text = self.extract_numbers_advanced(best_result, feature_name)
                
                # 特殊处理节段肌肉数据
                if "节段肌肉" in feature_name:
                    cleaned_text = self._process_muscle_data(cleaned_text, feature_name)
                
                logger.info(f"特征 {feature_name} 处理后结果: {cleaned_text}")
                
                # 最后验证: 确保结果为数字或为空
                if cleaned_text and not re.match(r'^[\d\.\-\+]+$', cleaned_text):
                    logger.warning(f"非数字结果被过滤: {cleaned_text}")
                    return ""
                
                return cleaned_text
            
            logger.warning(f"特征 {feature_name} 未能识别有效文本")
            return ""
            
        except Exception as e:
            logger.error(f"OCR识别错误: {str(e)}")
            return ""
    
    def _extract_ocr_result(self, result):
        """
        安全地从OCR结果中提取文本和置信度
        
        参数:
        result: OCR结果
        
        返回:
        (text, confidence) 元组
        """
        text = ""
        confidence = 0
        
        try:
            if not result or not isinstance(result, list) or len(result) == 0:
                return text, confidence

            # 统一处理不同版本PaddleOCR的返回格式
            first_item = result[0]
            if isinstance(first_item, list):
                # 老版本格式
                for line in first_item:
                    if isinstance(line, list) and len(line) >= 2:
                        if isinstance(line[1], list) and len(line[1]) >= 2:
                            text += str(line[1][0])
                            confidence = max(confidence, float(line[1][1]))
                        elif isinstance(line[1], tuple) and len(line[1]) >= 2:
                            text += str(line[1][0])
                            confidence = max(confidence, float(line[1][1]))
                        elif isinstance(line[1], str):
                            text += line[1]
                            confidence = max(confidence, 0.5)  # 默认置信度
            else:
                # 新版本格式
                for line in result:
                    if hasattr(line, 'boxes') and hasattr(line, 'rec_res'):
                        text_item, conf = line.rec_res
                        text += str(text_item)
                        confidence = max(confidence, float(conf))
        except (IndexError, TypeError, ValueError, AttributeError) as e:
            logger.debug(f"提取OCR结果时出错: {str(e)}")
        
        return text, confidence
    
    def _process_muscle_data(self, text, feature_name):
        """
        处理节段肌肉数据的特殊逻辑
        """
        if not text:
            return ""
        
        try:
            # 尝试直接转换
            value = float(text)
            
            # 调整异常值
            if value >= 1000:
                value = value / 1000
            elif value >= 100:
                value = value / 100
            elif value >= 50:
                value = value / 10
            
            # 确保在合理范围内 (节段肌肉通常在1-25范围)
            if "躯干" in feature_name:
                # 躯干肌肉通常较大，范围在10-30
                if value > 30:
                    value = value / 10
                elif value < 1:
                    value = value * 10
            else:
                # 肢体肌肉通常较小，范围在1-10
                if value > 20:
                    value = value / 10
                elif value < 0.5 and value > 0:
                    value = value * 10
            
            # 格式化为两位小数
            return f"{value:.2f}"
            
        except ValueError:
            # 如果转换失败，尝试提取数字
            digits = re.findall(r'\d+\.?\d*', text)
            if digits:
                try:
                    value = float(digits[0])
                    # 执行与上面相同的规范化逻辑
                    if value >= 100:
                        value = value / 100
                    elif value >= 50:
                        value = value / 10
                        
                    if "躯干" in feature_name:
                        if value > 30:
                            value = value / 10
                        elif value < 1:
                            value = value * 10
                    else:
                        if value > 20:
                            value = value / 10
                        elif value < 0.5 and value > 0:
                            value = value * 10
                        
                    return f"{value:.2f}"
                except ValueError:
                    pass
        
        return ""

def create_optimized_processor():
    """
    创建优化的OCR处理器实例
    """
    try:
        return OptimizedOCRProcessor(lang='ch', use_angle_cls=True)
    except Exception as e:
        logger.error(f"无法创建OCR处理器: {str(e)}")
        return None

if __name__ == "__main__":
    # 测试代码
    processor = create_optimized_processor()
    if processor:
        print("OCR处理器创建成功")
    else:
        print("OCR处理器创建失败") 