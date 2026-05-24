#!/usr/bin/env python3
import subprocess
import re
import os
import sys

def parse_log_output(log_content):
    """解析ST-SLAM日志输出，提取关键指标"""
    metrics = {
        'ate_rmse': None,
        'ate_mean': None,
        'rpe_trans_rmse': None,
        'rpe_rot_rmse': None,
        'loops_detected': None,
        'tracking_success': None,
        'num_keyframes': None,
        'total_frames': None,
        'wall_time': None,
        'edges_built': None,
        'loop_edges': None,
        'pgo_correction': None
    }
    
    # 解析ATE（支持多种格式）
    ate_match = re.search(r'--- ATE.*---.*RMSE:\s*([\d.]+)\s*m', log_content, re.DOTALL)
    if not ate_match:
        ate_match = re.search(r'ATE.*RMSE.*:\s*([\d.]+)\s*m', log_content)
    if not ate_match:
        ate_match = re.search(r'RMSE:\s*([\d.]+)\s*m', log_content)
    if ate_match:
        metrics['ate_rmse'] = float(ate_match.group(1))
    
    # 解析RPE Translation
    rpe_trans_match = re.search(r'Translation RMSE:\s*([\d.]+)\s*m', log_content)
    if rpe_trans_match:
        metrics['rpe_trans_rmse'] = float(rpe_trans_match.group(1))
    
    # 解析RPE Rotation
    rpe_rot_match = re.search(r'Rotation RMSE:\s*([\d.]+)\s*deg', log_content)
    if rpe_rot_match:
        metrics['rpe_rot_rmse'] = float(rpe_rot_match.group(1))
    
    # 解析回环检测数量
    loops_match = re.search(r'Loops Detected:\s*(\d+)', log_content)
    if loops_match:
        metrics['loops_detected'] = int(loops_match.group(1))
    
    # 解析跟踪成功率
    success_match = re.search(r'Tracking Success Rate:\s*([\d.]+)%', log_content)
    if success_match:
        metrics['tracking_success'] = float(success_match.group(1))
    
    # 解析关键帧数量
    kf_match = re.search(r'Running global pose graph optimization on (\d+) keyframes', log_content)
    if kf_match:
        metrics['num_keyframes'] = int(kf_match.group(1))
    
    # 解析总帧数
    frames_match = re.search(r'Total Frames:\s*(\d+)', log_content)
    if frames_match:
        metrics['total_frames'] = int(frames_match.group(1))
    
    # 解析运行时间
    wall_time_match = re.search(r'Wall time:\s*([\d.]+)\s*s', log_content)
    if wall_time_match:
        metrics['wall_time'] = float(wall_time_match.group(1))
    
    # 解析边数量和回环边数量
    edges_match = re.search(r'Built (\d+) edges \((\d+) odometry, (\d+) loop\)', log_content)
    if edges_match:
        metrics['edges_built'] = int(edges_match.group(1))
        metrics['loop_edges'] = int(edges_match.group(3))
    
    # 解析PGO校正量
    correction_match = re.search(r'PGO correction: trans=([\d.]+)m', log_content)
    if correction_match:
        metrics['pgo_correction'] = float(correction_match.group(1))
    
    return metrics

def print_metrics(metrics):
    """格式化打印关键指标"""
    print("="*60)
    print("          ST-SLAM 4.0 - 关键指标解析结果")
    print("="*60)
    
    print("\n📊 精度指标 (ATE)")
    print("-" * 30)
    if metrics['ate_rmse'] is not None:
        quality = "✨ SOTA" if metrics['ate_rmse'] < 0.01 else \
                  "✅ Excellent" if metrics['ate_rmse'] < 0.015 else \
                  "👍 Good" if metrics['ate_rmse'] < 0.03 else \
                  "⚠️ Acceptable" if metrics['ate_rmse'] < 0.06 else \
                  "❌ Needs Work"
        print(f"  ATE RMSE:    {metrics['ate_rmse']*100:.2f} cm  [{quality}]")
    else:
        print(f"  ATE RMSE:    N/A")
    
    print("\n📈 局部精度指标 (RPE)")
    print("-" * 30)
    if metrics['rpe_trans_rmse'] is not None:
        print(f"  Translation RMSE: {metrics['rpe_trans_rmse']*100:.2f} cm")
    if metrics['rpe_rot_rmse'] is not None:
        print(f"  Rotation RMSE:    {metrics['rpe_rot_rmse']:.2f} deg")
    
    print("\n🔄 回环检测")
    print("-" * 30)
    if metrics['loops_detected'] is not None:
        status = "✅ 回环检测成功！" if metrics['loops_detected'] > 0 else "❌ 未检测到回环"
        print(f"  Loops Detected:   {metrics['loops_detected']}  {status}")
    if metrics['loop_edges'] is not None:
        print(f"  Loop Edges in PGO: {metrics['loop_edges']}")
    if metrics['pgo_correction'] is not None:
        print(f"  PGO Correction:   {metrics['pgo_correction']*100:.2f} cm")
    
    print("\n📁 数据统计")
    print("-" * 30)
    if metrics['total_frames'] is not None:
        print(f"  Total Frames:     {metrics['total_frames']}")
    if metrics['num_keyframes'] is not None:
        print(f"  Keyframes:        {metrics['num_keyframes']}")
    if metrics['edges_built'] is not None:
        print(f"  PoseGraph Edges:  {metrics['edges_built']}")
    if metrics['tracking_success'] is not None:
        print(f"  Success Rate:     {metrics['tracking_success']:.1f}%")
    
    print("\n⏱️ 性能指标")
    print("-" * 30)
    if metrics['wall_time'] is not None:
        fps = metrics['total_frames'] / metrics['wall_time'] if metrics['total_frames'] else 0
        print(f"  Wall Time:        {metrics['wall_time']:.2f} s")
        print(f"  FPS:              {fps:.1f}")
    
    print("\n" + "="*60)
    
    # 总结评估
    print("\n📋 评估总结")
    print("-" * 30)
    if metrics['ate_rmse'] is not None and metrics['loops_detected'] is not None:
        if metrics['ate_rmse'] < 0.06 and metrics['loops_detected'] > 0:
            print("✅ 系统运行正常，回环检测生效！")
        elif metrics['loops_detected'] == 0:
            print("⚠️ 回环未检测到，请检查DBoW3集成和参数设置")
        else:
            print("⚠️ ATE较高，请检查位姿图优化配置")

def test_mode():
    """测试模式：使用模拟数据验证解析功能"""
    print("🧪 运行测试模式...")
    
    log_content = """
[DEBUG] New KF created: 6 keyframes total
[DEBUG] BuildFromKeyframes: kept 1 loop edges
[PGO] Built 6 edges (5 odometry, 1 loop)
[LOOP_CLOSED] KF1→KF5 drift=2.3cm 3.3deg
[DEBUG] PGO correction: trans=0.023m rot=3.3deg

ST-SLAM 4.0 ATE Quality:  [EXCELLENT]

 --- ATE (Absolute Trajectory Error) ---
   RMSE:                    0.012 m
   Mean:                    0.010 m

 --- RPE (Relative Pose Error) ---
   Translation RMSE:        0.008 m
   Rotation RMSE:           0.3 deg

 --- Loop Closure ---
   Loops Detected:          1

 --- Dataset & General ---
   Total Frames:            200
   Tracking Success Rate:    100.0%

   Wall time: 15.1s (0.08s/frame)
"""
    metrics = parse_log_output(log_content)
    print_metrics(metrics)

def main():
    if len(sys.argv) < 2:
        print("用法: python3 run_st_slam.py <数据集路径> [帧数]")
        print("示例: python3 run_st_slam.py /home/user/datasets/rgbd_dataset_freiburg1_desk 200")
        print("测试模式: python3 run_st_slam.py --test")
        sys.exit(1)
    
    # 测试模式
    if sys.argv[1] == '--test':
        test_mode()
        sys.exit(0)
    
    dataset_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    
    # 设置环境变量
    os.environ['LD_LIBRARY_PATH'] = '.'
    
    # 构建命令
    cmd = [
        './apps/st_slam_main',
        dataset_path,
        str(num_frames)
    ]
    
    print(f"🚀 正在运行 ST-SLAM 4.0...")
    print(f"📂 数据集: {dataset_path}")
    print(f"📦 帧数: {num_frames}")
    print("-" * 60)
    
    # 运行命令并捕获输出
    try:
        result = subprocess.run(
            cmd,
            cwd='/home/zhang/ORB_SLAM_Learning/st_slam/build',
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        # 打印原始输出（精简版）
        print("原始输出 (DEBUG信息):")
        print("-" * 30)
        for line in result.stdout.split('\n'):
            if '[DEBUG]' in line or '[PGO]' in line or '[LOOP_CLOSED]' in line:
                print(line)
        print("-" * 30)
        
        # 解析并打印指标
        metrics = parse_log_output(result.stdout + result.stderr)
        print_metrics(metrics)
        
        # 保存指标到文件
        with open('metrics_report.txt', 'w') as f:
            f.write("ST-SLAM 4.0 Metrics Report\n")
            f.write("="*50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        print("\n📝 指标已保存到 metrics_report.txt")
        
    except subprocess.TimeoutExpired:
        print("❌ 运行超时！")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ 找不到 st_slam_main，请确保已编译")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
