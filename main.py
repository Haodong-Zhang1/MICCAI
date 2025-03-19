# kits23_segmentation/main.py
import os
import argparse
import logging
import json
import sys

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from kits23_segmentation.data.preprocessing import KidneyTumorPreprocessor
from kits23_segmentation.utils.analysis import analyze_dataset, create_cv_splits
from kits23_segmentation.utils.visualization import visualize_data_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="KiTS23 Adaptive Multi-Scale Feature Fusion")

    # Dataset and output directories
    parser.add_argument("--dataset_dir", type=str, default="E:/MICCAI/dataset",
                        help="Path to KiTS23 dataset")
    parser.add_argument("--output_dir", type=str, default="E:/MICCAI/kits23_segmentation/output",
                        help="Output directory")

    # Task selection
    parser.add_argument("--preprocess", action="store_true", default=True,
                        help="Run preprocessing pipeline")
    parser.add_argument("--analyze", action="store_true", default=True,
                        help="Run dataset analysis")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualizations")

    # Preprocessing options
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of parallel workers for preprocessing (0=auto)")
    parser.add_argument("--window_width", type=int, default=500,
                        help="HU window width for preprocessing")
    parser.add_argument("--window_level", type=int, default=50,
                        help="HU window level for preprocessing")
    parser.add_argument("--crop_margin", type=int, default=30,
                        help="Margin for kidney region cropping")
    parser.add_argument("--skip_resample", action="store_true",
                        help="Skip resampling step in preprocessing")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Target voxel spacing for resampling")

    # Analysis options
    parser.add_argument("--cv_splits", type=int, default=5,
                        help="Number of cross-validation splits")
    parser.add_argument("--analyze_predictions", type=str, default=None,
                        help="Path to predictions directory for analysis")

    return parser.parse_args()


def setup_logging(output_dir):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'logs', 'main.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("KiTS23-Main")


def main():
    """Main execution function."""
    print("开始执行主函数...")
    args = parse_args()
    print(f"命令行参数: {vars(args)}")

    # 确保输出目录存在
    print("创建输出目录...")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'preprocessed_data'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    # Setup logging
    print("设置日志...")
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting KiTS23 Adaptive Multi-Scale Feature Fusion pipeline")
    logger.info(f"Output directory: {args.output_dir}")

    if args.analyze:
        print("开始数据集分析...")
        logger.info("Analyzing dataset...")
        metadata_df = analyze_dataset(args.dataset_dir, args.output_dir)
        cv_splits = create_cv_splits(
            metadata_df,
            n_splits=args.cv_splits,
            output_dir=args.output_dir
        )

        if args.visualize:
            print("生成数据集可视化...")
            logger.info("Generating dataset visualizations...")
            visualize_data_distribution(
                metadata_df,
                os.path.join(args.output_dir, "visualizations")
            )

    # Run preprocessing if requested
    if args.preprocess:
        print("开始数据预处理...")
        logger.info("Preprocessing dataset...")

        # Configure preprocessing steps based on command line arguments
        preprocessing_steps = [
            {'name': 'hu_window', 'params': {'window_width': args.window_width, 'window_level': args.window_level}},
            {'name': 'normalize', 'params': {'min_bound': -100, 'max_bound': 400}}
        ]

        if not args.skip_resample:
            preprocessing_steps.append(
                {'name': 'resample', 'params': {'target_spacing': tuple(args.target_spacing)}}
            )

        preprocessing_steps.append(
            {'name': 'kidney_region_crop', 'params': {'margin': args.crop_margin}}
        )

        print(f"预处理步骤: {preprocessing_steps}")

        # Create and configure preprocessor
        preprocessor = KidneyTumorPreprocessor(
            output_dir=os.path.join(args.output_dir, "preprocessed_data"),
            preprocessing_steps=preprocessing_steps
        )

        # Run preprocessing
        processed_cases = preprocessor.preprocess_dataset(
            args.dataset_dir,
            num_workers=args.num_workers
        )

        print(f"预处理完成，处理了 {len(processed_cases)} 个病例")

    print("程序执行完成！")

if __name__ == "__main__":
    try:
        print("程序启动...")
        print("当前工作目录:", os.getcwd())
        print("Python路径:", sys.path)
        main()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()